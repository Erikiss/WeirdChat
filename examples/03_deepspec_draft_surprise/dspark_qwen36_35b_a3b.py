"""DeepSpec DSpark config for the WeirdChat Qwen subject model (qwen3.6-35b-a3b).

This file lives outside the DeepSpec tree and is passed by path, which
DeepSpec's ``config_path`` mechanism supports:

    # phase 2 (from $DEEPSPEC_ROOT):
    python scripts/data/prepare_target_cache.py \
        --config /path/to/dspark_qwen36_35b_a3b.py \
        --train-data-path .../baseline_train.jsonl \
        --output-dir ~/.cache/deepspec/qwen36_35b_a3b_target_cache

    # phase 3:
    bash scripts/train/train.sh   # with config_path pointed here

Layer geometry (``model.target_layer_ids``) defaults to a 48-layer guess; run
phase0_feasibility.py first and override with its recommendation:
    --opts model.target_layer_ids="[...]"

DeepSpec loads config modules with ``exec_module`` in every spawned worker, so
registering the WeirdChat chat template here guarantees it is present wherever
``data.chat_template`` is resolved.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from surprise_common import (
    WEIRDCHAT_TEMPLATE_NAME,
    detect_assistant_prefix,
    register_weirdchat_template,
)

# Optionally strip the qwen3.6 <think></think> scaffold from the assistant loss
# mask (WEIRDSPEC_STRIP_THINK=1). Off by default: the default rendering is what
# phase 0's parser gate validated.
_strip_prefix = None
if os.environ.get("WEIRDSPEC_STRIP_THINK") == "1":
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(
        os.environ.get("WEIRDSPEC_TARGET_MODEL", "Qwen/Qwen3.6-35B-A3B")
    )
    _strip_prefix = detect_assistant_prefix(_tok) or None

register_weirdchat_template(strip_think_prefix=_strip_prefix)

from deepspec.trainer import Qwen3DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR

project_name = "weirdspec"
exp_name = "dspark_block7_qwen36_35b_a3b"
seed = 42

model = dict(
    target_model_name_or_path=os.environ.get("WEIRDSPEC_TARGET_MODEL", "Qwen/Qwen3.6-35B-A3B"),
    block_size=7,
    num_draft_layers=5,
    # Placeholder for a 48-layer target — override with phase 0's
    # recommended_target_layer_ids via --opts.
    target_layer_ids=[1, 12, 23, 34, 45],
    # The dense draft needs an intermediate_size; the MoE target defines none,
    # so set it from phase 0's recommended_draft_intermediate_size.
    draft_intermediate_size=int(os.environ.get("WEIRDSPEC_DRAFT_INTERMEDIATE_SIZE", "0")) or None,
    # DeepSpec's Qwen3 default (151669, `<|fim_pad|>`) is NOT a special token
    # in the qwen3.6 tokenizer — set this from phase 0's
    # recommended_mask_token_id.
    mask_token_id=int(os.environ.get("WEIRDSPEC_MASK_TOKEN_ID", "151669")),
    num_anchors=512,

    ## markov head
    markov_rank=256,
    markov_head_type="vanilla",

    ## confidence head — required: phase 4 scoring consumes its predictions.
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    ## loss
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=Qwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    # WeirdChat protocol: the registered template injects no system prompt.
    chat_template=WEIRDCHAT_TEMPLATE_NAME,
    # WeirdChat sampled at most 1024 new tokens per turn; 4096 leaves room for
    # multi-turn prompts while keeping the cache bounded.
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project = str(cfg["project_name"])
    experiment = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project, experiment)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project, experiment)
    cfg["logging"] = logging_cfg
    return cfg
