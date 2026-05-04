#!/bin/zsh

# Convenience script to run a specific pipeline stage.
# Usage: ./scripts/run_stage.sh <stage_name>
# Example: ./scripts/run_stage.sh assessment

STAGE="${1:-preprocess}"
PROFILE="${2:-custom}"
CLIENT_PROFILE="${3:-gemini-paid}"
PAPER_PATH="${4:-resources/papers/0400.pdf}"
PROMPT_PATH="${5:-resources/prompts_master.yaml}"
PROMPT_KEY="${6:-detailed-csv}"
GROUND_TRUTH_PATH="${7:-resources/correct_answers.csv}"

uv run rrp run-stage $STAGE \
    --profile $PROFILE \
    --client-profile $CLIENT_PROFILE \
    --paper-path $PAPER_PATH \
    --prompt-path $PROMPT_PATH \
    --prompt-key $PROMPT_KEY \
    --ground-truth-path $GROUND_TRUTH_PATH
