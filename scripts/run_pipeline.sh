#!/bin/zsh

# Convenience script to run the full research pipeline.
# Usage: ./scripts/run_pipeline.sh

PROFILE="${1:-custom}"
CLIENT_PROFILE="${2:-gemini-paid}"
PAPER_PATH="${3:-resources/papers/0400.pdf}"
PROMPT_PATH="${4:-resources/prompts_master.yaml}"
PROMPT_KEY="${5:-detailed-csv}"
GROUND_TRUTH_PATH="${6:-resources/correct_answers.csv}"

uv run rrp run-pipeline \
    --profile $PROFILE \
    --client-profile $CLIENT_PROFILE \
    --paper-path $PAPER_PATH \
    --prompt-path $PROMPT_PATH \
    --prompt-key $PROMPT_KEY \
    --ground-truth-path $GROUND_TRUTH_PATH
