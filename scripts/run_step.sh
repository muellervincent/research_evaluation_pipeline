#!/bin/zsh

# Convenience script to run a specific pipeline step.
# Usage: ./scripts/run_step.sh <stage_name> <step_name>
# Example: ./scripts/run_step.sh assessment extract

STAGE="${1:-preprocess}"
STEP="${2:-refine}"
PROFILE="${3:-standard}"
PAPER_PATH="${4:-resources/papers/0400.pdf}"
PROMPT_PATH="${5:-resources/prompts_master.yaml}"
PROMPT_KEY="${6:-detailed-csv}"
GROUND_TRUTH_PATH="${7:-resources/correct_answers.csv}"

uv run rrp run-step $STAGE $STEP \
    --profile $PROFILE \
    --paper-path $PAPER_PATH \
    --prompt-path $PROMPT_PATH \
    --prompt-key $PROMPT_KEY \
    --ground-truth-path $GROUND_TRUTH_PATH
