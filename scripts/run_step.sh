#!/bin/zsh

# Convenience script to run a specific pipeline step.
# Usage: ./scripts/run_step.sh <stage_name> <step_name>
# Example: ./scripts/run_step.sh assessment extract

STAGE="${1:-preprocess}"
STEP="${2:-refine}"
PROFILE="${3:-standard}"
CLIENT_PROFILE="${4:-gemini-paid}"
PAPER_PATH="${5:-resources/papers/0400.pdf}"
PROMPT_PATH="${6:-resources/prompts_master.yaml}"
PROMPT_KEY="${7:-detailed-csv}"
GROUND_TRUTH_PATH="${8:-resources/correct_answers.csv}"

uv run rrp run-step $STAGE $STEP \
    --profile $PROFILE \
    --client-profile $CLIENT_PROFILE \
    --paper-path $PAPER_PATH \
    --prompt-path $PROMPT_PATH \
    --prompt-key $PROMPT_KEY \
    --ground-truth-path $GROUND_TRUTH_PATH
