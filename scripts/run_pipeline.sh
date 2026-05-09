#!/bin/zsh

# Convenience script to run the full research pipeline.
# Usage: ./scripts/run_pipeline.sh

PROFILE="${1:-hybrid}"
PAPER_PATH="${2:-resources/papers/0400.pdf}"
PROMPT_PATH="${3:-resources/prompts_master.yaml}"
PROMPT_KEY="${4:-detailed-csv}"
GROUND_TRUTH_PATH="${5:-resources/correct_answers.csv}"

uv run rrp run-pipeline \
    --profile $PROFILE \
    --paper-path $PAPER_PATH \
    --prompt-path $PROMPT_PATH \
    --prompt-key $PROMPT_KEY \
    --ground-truth-path $GROUND_TRUTH_PATH
