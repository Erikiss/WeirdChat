"""Client library for the WeirdChat dataset of unexpected AI behaviors.

Quick tour::

    import weirdchat as wc

    wc.behaviors()                                  # every behavior with its judge rubrics
    top = wc.patterns(subject_model="thinkingmachines/inkling")[0]
    wc.prompts(top)                                 # the pattern's measured prompts
    wc.transcripts(top)                             # its judged rollouts
    wc.RubricJudge.for_behavior(wc.behavior(top.behavior_id))  # the reference judge

See ``examples/01_quickstart`` for a walkthrough, including reproducing a behavior on OpenRouter.
"""

from .concurrency import run_bounded
from .dataset import (
    REPO_ID,
    WeirdChat,
    behavior,
    behaviors,
    default_client,
    highlight_transcript,
    pairwise_judgments,
    pattern,
    patterns,
    prompts,
    search_compute,
    transcripts,
)
from .judge import JudgeError, JudgeResult, RubricJudge
from .types import (
    Behavior,
    Citation,
    Elo,
    EloAxis,
    EloScore,
    Judgment,
    MatchSummary,
    Message,
    OpenRouterReplication,
    PairwiseJudgment,
    Pattern,
    PatternMetrics,
    Prompt,
    Role,
    Rubric,
    SearchCompute,
    Transcript,
)

__all__ = [
    "REPO_ID",
    "WeirdChat",
    "behavior",
    "behaviors",
    "default_client",
    "highlight_transcript",
    "pairwise_judgments",
    "pattern",
    "patterns",
    "prompts",
    "search_compute",
    "transcripts",
    "run_bounded",
    "JudgeError",
    "JudgeResult",
    "RubricJudge",
    "Behavior",
    "Citation",
    "Elo",
    "EloAxis",
    "EloScore",
    "Judgment",
    "MatchSummary",
    "Message",
    "OpenRouterReplication",
    "PairwiseJudgment",
    "Pattern",
    "PatternMetrics",
    "Prompt",
    "Role",
    "Rubric",
    "SearchCompute",
    "Transcript",
]
