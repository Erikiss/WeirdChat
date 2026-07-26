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
    pick_mask_token,
    recommend_target_layer_ids,
    register_weirdchat_template,
    resolve_deepspec_root,
    unwrap_text_config,
)


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

    # G2: the (possibly text_config-nested) config carries every field
    # DeepSpec's dense draft build needs.
    def g2() -> dict[str, Any]:
        cfg, nesting = unwrap_text_config(state["config"])
        state["effective_config"] = cfg
        report["config_nesting"] = nesting
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
        assert not missing, (
            f"target config (nesting={nesting}) lacks fields needed by the dense draft: {missing}"
        )
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
        if nesting is not None:
            report["needs_deepspec_patch"] = True
        return {
            "config_nesting": nesting,
            "num_hidden_layers": num_layers,
            "hidden_size": int(cfg.hidden_size),
            "recommended_target_layer_ids": rec,
            "target_is_moe": is_moe,
        }

    ok &= gate(report, "G2_draft_config_fields", True, g2)

    # G3: tokenizer loads and an unused special token can serve as the draft
    # mask token (DeepSpec's Qwen3 default 151669 no longer exists in 3.6).
    def g3() -> dict[str, Any]:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.target)
        state["tokenizer"] = tok
        specials = {
            str(added): int(idx)
            for idx, added in tok.added_tokens_decoder.items()
            if getattr(added, "special", False)
        }
        sample = tok.apply_chat_template(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ho"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        reserved = {t for t in specials if t in sample}
        for role_token in (tok.eos_token, tok.bos_token, tok.pad_token, tok.unk_token):
            if role_token:
                reserved.add(str(role_token))
        name, token_id = pick_mask_token(specials, reserved)
        report["recommended_mask_token"] = name
        report["recommended_mask_token_id"] = token_id
        return {
            "recommended_mask_token": name,
            "recommended_mask_token_id": token_id,
            "n_special_tokens": len(specials),
        }

    ok &= gate(report, "G3_mask_token", True, g3)

    # G4: chat-template rendering matches the WeirdChat protocol: no injected
    # system prompt, no <think> blocks (retrying with enable_thinking=False —
    # the patched parser passes that flag through), ChatML headers present.
    def g4() -> dict[str, Any]:
        tok = state["tokenizer"]
        convo = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        text = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        assert "You are a helpful assistant" not in text, "template injects a system prompt"
        needs_flag = "<think>" in text
        if needs_flag:
            text = tok.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            assert "<think>" not in text, (
                "template inserts think blocks even with enable_thinking=False"
            )
            report["requires_enable_thinking_false"] = True
            report["needs_deepspec_patch"] = True
        assert "<|im_start|>assistant" in text, "expected ChatML assistant header"
        return {"rendered_chars": len(text), "requires_enable_thinking_false": needs_flag}

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

    # G6 (--deep): meta-device instantiation of target and draft. Exercises the
    # patched build_draft_config path — apply deepspec_qwen36.patch first when
    # the target config is nested.
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
                    "mask_token_id": report["recommended_mask_token_id"],
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
    if report.get("needs_deepspec_patch"):
        print(
            "NOTE: this target needs deepspec_qwen36.patch applied to the DeepSpec "
            "checkout before phases 2-4 (cd $DEEPSPEC_ROOT && git apply "
            "/path/to/deepspec_qwen36.patch)."
        )
    if "recommended_mask_token_id" in report:
        print(
            "For phases 2-3, export WEIRDSPEC_MASK_TOKEN_ID="
            f"{report['recommended_mask_token_id']} "
            f"({report.get('recommended_mask_token')}) and pass "
            f"--opts model.target_layer_ids=\"{report.get('recommended_target_layer_ids')}\"."
        )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
