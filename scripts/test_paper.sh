#!/bin/bash
# test_paper.sh
# Run evaluation on a single paper with a chosen prompt and profile
# Usage: ./scripts/test_paper.sh <stem> <prompt_path> <profile>

set -e
cd "$(dirname "$0")/.."

PAPER_STEM=${1:-"0400"}
PROMPT_FILE=${2:-"resources/prompt_detailed.md"}
PROFILE=${3:-"paid_sota"}
MODE=${4:-"planning"}
INGESTION=${5:-"pdf"}
PLANNING_ARCH=${6:-"sequential"}
ANALYZE_MISMATCHES=${7:-"--analyze-mismatches"}

echo "====================================="
echo "RRP Single Paper Evaluation"
echo "====================================="
echo "Paper Stem:     $PAPER_STEM"
echo "Prompt File:    $PROMPT_FILE"
echo "Profile:        $PROFILE"
echo "Mode:           $MODE"
echo "Ingestion:      $INGESTION"
echo "Planning Arch:  $PLANNING_ARCH"
echo "Analyze Misc:   $ANALYZE_MISMATCHES"
echo "====================================="

PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training \
    --subset "$PAPER_STEM" \
    --prompt "$PROMPT_FILE" \
    --profile "$PROFILE" \
    --mode "$MODE" \
    --ingestion "$INGESTION" \
    --planning-arch "$PLANNING_ARCH" \
    $ANALYZE_MISMATCHES

