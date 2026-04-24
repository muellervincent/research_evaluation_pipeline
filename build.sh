#!/bin/bash

# Configuration defaults
PAPER_DIR="/Users/vincentmueller/Developer/Data/study_evalutation/papers/training"
FILE=""
MODE="fast"
STRATEGY="batch"
INTERACTIVE=""
SAMPLE_SIZE=""

# Help text
show_help() {
    echo "Usage: ./build.sh [OPTIONS] [FILE_PATH]"
    echo ""
    echo "Convenience wrapper for running the evaluation pipeline."
    echo ""
    echo "Options:"
    echo "  --fast         Run only Fast Mode (default)"
    echo "  --plan         Run only Planning Mode"
    echo "  --both         Run both Fast and Planning Modes and Auto-compare"
    echo "  --isolated     Use the isolated evaluation strategy (Per-Task)"
    echo "  --batch        Use the batch evaluation strategy (Summed Up - default)"
    echo "  --interactive  Enable interactive step-by-step confirmation prompts"
    echo "  --sample N     Test on N randomly sampled files from the directory"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./build.sh --both --isolated              # Tests a random paper in BOTH modes, isolated"
    echo "  ./build.sh --plan --interactive path/doc.pdf"
    echo "  ./build.sh --both --sample 3              # Tests 3 random papers sequentially"
    exit 0
}

# Parse custom flags
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --fast) MODE="fast"; shift ;;
        --plan) MODE="planning"; shift ;;
        --both) MODE="both"; shift ;;
        --isolated) STRATEGY="isolated"; shift ;;
        --batch) STRATEGY="batch"; shift ;;
        --interactive) INTERACTIVE="--interactive"; shift ;;
        --sample) SAMPLE_SIZE="--sample-size $2"; shift 2 ;;
        -h|--help) show_help ;;
        *) 
            if [ -z "$FILE" ]; then
                FILE="$1"
            else
                echo "Unknown option or multiple files provided: $1"
                exit 1
            fi
            shift ;;
    esac
done

# Resolve Target File/Directory
if [ -n "$SAMPLE_SIZE" ]; then
    # If sampling, target the directory instead of a single file
    TARGET="$PAPER_DIR"
    echo "=========================================================="
    echo "Batch Testing: Sampling papers from $TARGET"
    echo "Mode: $MODE | Strategy: $STRATEGY"
    echo "=========================================================="
else
    if [ -z "$FILE" ]; then
        # Cross-platform way to get a random file
        TARGET=$(find "$PAPER_DIR" -name "*.pdf" | sort -R | head -n 1)
        echo "No file provided. Randomly selected: $(basename "$TARGET")"
    else
        TARGET="$FILE"
    fi

    if [ ! -f "$TARGET" ]; then
        echo "Error: File not found at $TARGET"
        exit 1
    fi
    
    echo "=========================================================="
    echo "Testing file: $(basename "$TARGET")"
    echo "Mode: $MODE | Strategy: $STRATEGY"
    echo "=========================================================="
fi

# Construct and run the command
CMD="uv run python main.py --target \"$TARGET\" --mode $MODE --strategy $STRATEGY $INTERACTIVE $SAMPLE_SIZE"

echo "Executing: $CMD"
echo ""

eval $CMD
