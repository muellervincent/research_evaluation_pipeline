#!/bin/bash

# Directory containing the papers
PAPER_DIR="/Users/vincentmueller/Developer/Data/study_evalutation/papers/training"

# Find a random file if no argument is provided
if [ -z "$1" ]; then
    # Cross-platform way to get a random file (works on macOS and Linux)
    FILE=$(find "$PAPER_DIR" -name "*.pdf" | sort -R | head -n 1)
    echo "No file provided. Randomly selected: $FILE"
else
    FILE="$1"
fi

if [ ! -f "$FILE" ]; then
    echo "File not found: $FILE"
    exit 1
fi

echo "=========================================================="
echo "Testing file: $FILE"
echo "=========================================================="

# Run the fast, single-pass evaluation
uv run python main.py "$FILE" --mode fast

# Run the complex, evidence-grounded planning evaluation
uv run python main.py "$FILE" --mode planning
