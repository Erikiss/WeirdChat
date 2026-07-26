"""Phase 0 — feasibility gates for running DeepSpec against qwen3.6-35b-a3b.

Runs cheap, mostly-CPU checks and writes ``phase0_report.json`` with the
discovered model geometry and the recommended ``target_layer_ids``. Every gate
records pass/fail plus detail; the script exits non-zero if any hard gate
fails, so it can guard the rest of the pipeline in automation.

Usage:
    python phase0_feasibility.py --deepspec-root /path/to/deepspec \
        [--target Qwen/Qwen3.6-35B-A3B] [--deep]

``--deep`` additionally instantiates the target and the draft model on the
meta device (no weight download, but imports the full modeling stack).
"""

from __future__ import annotations

import argparse
import json
import traceback
from typing import Any, Callable

from surprise_common import (
    DEFAULT_TARGET_MODEL,
    recommend_target_layer_ids,
    register_weirdchat_template,
    resolve_deepspec_root,
)

# DeepSpec's shipped Qwen configs use this id (an unused Qwen special token) as
# the draft mask token. Gate G3 verifies it exists in the 3.6 tokenizer.
CANDIDATE_MASK_TOKEN_ID = 151669


def gate(report: dict[str, Any], name: str, hard: bool, fn: Callable[[], dict[str, Any]]) -> bool:
    try:
        detail = fn()
        report["gates"][name] = {"passed": True, "hard": hard, **detail}
        print(f"PASS {name}: {json.dumps(detail, default=str)[:300]}")
        return True
    except Exception as e:  # noqa: BLE001 — every gate failure must be reported, not raised
        report["gates"][name] = {
            "passed": False,
            "hard": hard,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
        }
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        return not hard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspec-root", default=None)
    parser.add_argument("--target", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--output", default="phase0_report.json")
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()

    resolve_deepspec_root(args.deepspec_root)

    report: dict[str, Any] = {"target": args.target, "gates": {}}
    ok = True

    # G1: transformers knows the architecture and AutoConfig loads.
    state: dict[str, Any] = {}

    def g1() -> dict[str, Any]:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(args.target)
        state["config"] = cfg
        return {"model_type": cfg.model_type, "architectures": getattr(cfg, "architectures", None)}

    ok &= gate(report, "G1_autoconfig", True, g1)

    # G2: the config carries every field DeepSpec's dense draft build needs.
    def g2() -> dict[str, Any]:
        cfg = state["config"]
        required = [
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "vocab_size",
        ]
        missing = [f for f in required if getattr(cfg, f, None) is None]
        assert not missing, f"target config lacks fields needed by the dense draft: {missing}"
        num_layers = int(cfg.num_hidden_layers)
        rec = recommend_target_layer_ids(num_layers)
        report["num_hidden_layers"] = num_layers
        report["hidden_size"] = int(cfg.hidden_size)
        report["recommended_target_layer_ids"] = rec
        is_moe = any(
            getattr(cfg, f, None) is not None
            for f in ("num_experts", "num_routed_experts", "moe_intermediate_size")
        )
        report["target_is_moe"] = is_moe
        return {
            "num_hidden_layers": num_layers,
            "hidden_size": int(cfg.hidden_size),
            "recommended_target_layer_ids": rec,
            "target_is_moe": is_moe,
        }

    ok &= gate(report, "G2_draft_config_fields", True, g2)

    # G3: tokenizer loads and the candidate mask token is a real special token.
    def g3() -> dict[str, Any]:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.target)
        state["tokenizer"] = tok
        piece = tok.convert_ids_to_tokens(CANDIDATE_MASK_TOKEN_ID)
        assert piece is not None, f"token id {CANDIDATE_MASK_TOKEN_ID} not in vocab"
        looks_special = str(piece).startswith("<|")
        report["mask_token_id"] = CANDIDATE_MASK_TOKEN_ID
        report["mask_token_piece"] = str(piece)
        assert looks_special, (
            f"id {CANDIDATE_MASK_TOKEN_ID} decodes to {piece!r}, which does not look like an "
            "unused special token — pick a different mask_token_id for the config"
        )
        return {"mask_token_id": CANDIDATE_MASK_TOKEN_ID, "piece": str(piece)}

    ok &= gate(report, "G3_mask_token", True, g3)

    # G4: chat-template rendering matches the WeirdChat protocol: no injected
    # system prompt, no <think> blocks, ChatML assistant headers present.
    def g4() -> dict[str, Any]:
        tok = state["tokenizer"]
        convo = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        text = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        assert "You are a helpful assistant" not in text, "template injects a system prompt"
        assert "<think>" not in text, "template inserts think blocks into plain transcripts"
        assert "<|im_start|>assistant" in text, "expected ChatML assistant header"
        return {"rendered_chars": len(text)}

    ok &= gate(report, "G4_chat_template", True, g4)

    # G5: DeepSpec's parser produces a non-empty assistant loss mask with the
    # registered WeirdChat template.
    def g5() -> dict[str, Any]:
        register_weirdchat_template()
        from deepspec.data.parser import preprocess_record  # type: ignore[import-not-found]

        from surprise_common import WEIRDCHAT_TEMPLATE_NAME

        record = {
            "conversations": [
                {"role": "user", "content": "Say something."},
                {"role": "assistant", "content": "Something, as requested."},
            ]
        }
        out = preprocess_record(record, state["tokenizer"], WEIRDCHAT_TEMPLATE_NAME, 512)
        n_loss = int(out["loss_mask"].sum())
        assert n_loss > 0, "assistant loss mask is empty — template/regex mismatch"
        return {"loss_tokens": n_loss, "total_tokens": int(out["attention_mask"].sum())}

    ok &= gate(report, "G5_parser_loss_mask", True, g5)

    # G6 (--deep): meta-device instantiation of target and draft.
    if args.deep:

        def g6() -> dict[str, Any]:
            import torch
            from transformers import AutoModel

            from deepspec.modeling.dspark.qwen3.config import (  # type: ignore[import-not-found]
                build_draft_config,
            )
            from deepspec.modeling.dspark.qwen3 import (  # type: ignore[import-not-found]
                Qwen3DSparkModel,
            )
            from deepspec.utils.config import to_config_node  # type: ignore[import-not-found]

            cfg = state["config"]
            with torch.device("meta"):
                AutoModel.from_config(cfg)
            model_args = to_config_node(
                {
                    "num_draft_layers": 5,
                    "target_layer_ids": report["recommended_target_layer_ids"],
                    "block_size": 7,
                    "mask_token_id": CANDIDATE_MASK_TOKEN_ID,
                    "num_anchors": 512,
                    "markov_rank": 256,
                    "markov_head_type": "vanilla",
                    "confidence_head_alpha": 1.0,
                    "confidence_head_with_markov": True,
                }
            )
            draft_cfg = build_draft_config(cfg, model_args)
            with torch.device("meta"):
                draft = Qwen3DSparkModel(draft_cfg)
            n_params = sum(p.numel() for p in draft.parameters())
            return {"draft_params": n_params}

        ok &= gate(report, "G6_meta_instantiation", True, g6)

    report["all_hard_gates_passed"] = ok
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written to {args.output}. All hard gates passed: {ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
