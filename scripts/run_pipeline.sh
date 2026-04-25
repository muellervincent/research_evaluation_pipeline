#!/bin/bash
# Convenient execution pipeline recipes for RRP Evaluation
# Automatically switches to project root

set -e
cd "$(dirname "$0")/.."

echo "====================================="
echo " RRP Evaluation Pipeline Execution"
echo "====================================="
echo "1) Both Modes on Unpaid Lite"
echo "2) Both Modes on Paid Lite"
echo "3) Both Modes on Unpaid Standard"
echo "4) Both Modes on Paid Standard"
echo "5) Both Modes on Unpaid SOTA"
echo "6) Both Modes on Paid SOTA"
echo "====================================="
echo "7) Planning Mode only on Unpaid Lite"
echo "8) Planning Mode only on Paid Lite"
echo "9) Planning Mode only on Unpaid Standard"
echo "10) Planning Mode only on Paid Standard"
echo "11) Planning Mode only on Unpaid SOTA"
echo "12) Planning Mode only on Paid SOTA"
echo "====================================="
echo "13) Fast Mode only on Unpaid Lite"
echo "14) Fast Mode only on Paid Lite"
echo "15) Fast Mode only on Unpaid Standard"
echo "16) Fast Mode only on Paid Standard"
echo "17) Fast Mode only on Unpaid SOTA"
echo "18) Fast Mode only on Paid SOTA"
echo "====================================="
read -p "Enter choice [1-18]: " choice

SUBSET="0191,0646,0665,0214,0400"

case $choice in
    1)
        echo "Running Both Modes on Unpaid Lite..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode both --profile unpaid_lite --analyze-mismatches
        ;;
    2)
        echo "Running Both Modes on Paid Lite..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode both --profile paid_lite --analyze-mismatches
        ;;
    3)
        echo "Running Both Modes on Unpaid Standard..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode both --profile unpaid_standard --analyze-mismatches
        ;;
    4)      
        echo "Running Both Modes on Paid Standard..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode both --profile paid_standard --analyze-mismatches
        ;;
    5)
        echo "Running Both Modes on Unpaid SOTA..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode both --profile unpaid_sota --analyze-mismatches
        ;;
    6)
        echo "Running Both Modes on Paid SOTA..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode both --profile paid_sota --analyze-mismatches
        ;;
    7)
        echo "Running Planning Mode only on Unpaid Lite..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode planning --profile unpaid_lite --analyze-mismatches
        ;;
    8)
        echo "Running Planning Mode only on Paid Lite..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode planning --profile paid_lite --analyze-mismatches
        ;;
    9)
        echo "Running Planning Mode only on Unpaid Standard..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode planning --profile unpaid_standard --analyze-mismatches
        ;;
    10)
        echo "Running Planning Mode only on Paid Standard..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode planning --profile paid_standard --analyze-mismatches
        ;;
    11)
        echo "Running Planning Mode only on Unpaid SOTA..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode planning --profile unpaid_sota --analyze-mismatches
        ;;
    12)
        echo "Running Planning Mode only on Paid SOTA..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode planning --profile paid_sota --analyze-mismatches
        ;;
    13)
        echo "Running Fast Mode only on Unpaid Lite..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode fast --profile unpaid_lite --analyze-mismatches
        ;;
    14)
        echo "Running Fast Mode only on Paid Lite..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode fast --profile paid_lite --analyze-mismatches
        ;;
    15)
        echo "Running Fast Mode only on Unpaid Standard..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode fast --profile unpaid_standard --analyze-mismatches
        ;;
    16)
        echo "Running Fast Mode only on Paid Standard..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode fast --profile paid_standard --analyze-mismatches
        ;;
    17)
        echo "Running Fast Mode only on Unpaid SOTA..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode fast --profile unpaid_sota --analyze-mismatches
        ;;
    18)
        echo "Running Fast Mode only on Paid SOTA..."
        PYTHONPATH=src uv run python -m rrp_eval.cli evaluate resources/papers/training --subset "$SUBSET" --mode fast --profile paid_sota --analyze-mismatches
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
