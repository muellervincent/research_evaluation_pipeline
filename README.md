# Research Evaluation Pipeline

This pipeline automates the evaluation and diagnostic analysis of research papers using Large Language Models (LLMs). It manages document ingestion, criteria-based assessment, and root-cause analysis of discrepancies between model outputs and ground truth.

## Execution Architecture

This diagram shows the pipeline's data flow, execution architecture, and the artifacts generated at each stage.

```mermaid
graph TD
    %% Global Inputs
    PDF_FILE(["Source PDF File"])
    MASTER_PROMPT(["Master Prompt (YAML/MD)"])
    GROUND_TRUTH(["Ground Truth (CSV)"])

    %% Paper Context Lifecycle
    subgraph CONTEXT_LIFECYCLE["Document Lifecycle"]
        direction TB
        PC_INIT["Step: Context Initialization"]
        PAPER_CONTEXT[/"ARTIFACT: Paper Context (PDF + Extracted MD + IDs)"/]
        
        PDF_FILE --> PC_INIT
        PC_INIT --> PAPER_CONTEXT
    end

    %% Stage 1: Preprocess
    subgraph PREPROCESS["Stage 1: Preprocess"]
        direction TB
        S1_REFINE["Step: Refine Prompt"]
        S1_EXTRACT["Step: MD Extraction (Mode: MD)"]
        
        REFINED_PROMPT["ARTIFACT: Refined Prompt"]

        S1_REFINE --> REFINED_PROMPT
        S1_EXTRACT --> PAPER_CONTEXT
    end

    MASTER_PROMPT --> S1_REFINE
    PAPER_CONTEXT --> S1_EXTRACT

    %% Stage 2: Assessment
    subgraph ASSESSMENT["Stage 2: Assessment"]
        direction TB
        S2_FAST["Step: Fast Assessment"]
        S2_DECOMP["Step: Criteria Decomposition"]
        S2_EXTRACT["Step: Evidence Extraction"]
        S2_SYNTH["Step: Group Synthesis"]
        
        ASS_TASK_LIST["ARTIFACT: Assessment Task List"]
        EVIDENCE_REPORTS["ARTIFACT: Evidence Reports"]
        ASSESSMENT_REPORT["ARTIFACT: Assessment Report"]

        S2_DECOMP --> ASS_TASK_LIST
        ASS_TASK_LIST --> S2_EXTRACT
        S2_EXTRACT --> EVIDENCE_REPORTS
        EVIDENCE_REPORTS --> S2_SYNTH
        S2_SYNTH --> ASSESSMENT_REPORT
        S2_FAST --> ASSESSMENT_REPORT
    end

    REFINED_PROMPT --> S2_FAST
    REFINED_PROMPT --> S2_DECOMP
    PAPER_CONTEXT --> S2_FAST
    PAPER_CONTEXT --> S2_EXTRACT

    %% Stage 3: Diagnostic
    subgraph DIAGNOSTIC["Stage 3: Diagnostic"]
        direction TB
        S3_FILTER["Internal: Prediction Filtering"]
        S3_FAST["Step: Fast Diagnostic"]
        S3_DECOMP["Step: Diagnostic Decomposition"]
        S3_ANALYZE["Step: Logic Analysis"]
        
        FILTERED_PREDICTIONS["ARTIFACT: Diagnostic Target List"]
        DIAG_PROMPT_INPUT["ARTIFACT: Selected Diagnostic Prompt"]
        DIAG_TASK_LIST["ARTIFACT: Diagnostic Task List"]
        DIAGNOSTIC_REPORT["ARTIFACT: Diagnostic Report"]

        S3_FILTER --> FILTERED_PREDICTIONS
        S3_DECOMP --> DIAG_TASK_LIST
        DIAG_TASK_LIST --> S3_ANALYZE
        S3_ANALYZE --> DIAGNOSTIC_REPORT
        S3_FAST --> DIAGNOSTIC_REPORT
    end

    ASSESSMENT_REPORT --> S3_FILTER
    GROUND_TRUTH --> S3_FILTER

    MASTER_PROMPT --> DIAG_PROMPT_INPUT
    REFINED_PROMPT --> DIAG_PROMPT_INPUT

    DIAG_PROMPT_INPUT --> S3_FAST
    DIAG_PROMPT_INPUT --> S3_DECOMP
    DIAG_PROMPT_INPUT --> S3_ANALYZE

    FILTERED_PREDICTIONS --> S3_FAST
    FILTERED_PREDICTIONS --> S3_DECOMP
    FILTERED_PREDICTIONS --> S3_ANALYZE

    PAPER_CONTEXT --> S3_FAST
    PAPER_CONTEXT --> S3_ANALYZE

    %% Stage 4: Results
    subgraph RESULTS["Stage 4: Result Generation"]
        direction TB
        S4_MERGE["Step: Artifact Reconstruction"]
        S4_GEN_MD["Step: Markdown Generator"]
        S4_GEN_JSON["Step: JSON Generator"]
        
        FINAL_DATA_MODEL["ARTIFACT: Final Result Model"]
        MD_OUTPUT["OUTPUT: report.md"]
        JSON_OUTPUT["OUTPUT: result.json"]

        S4_MERGE --> FINAL_DATA_MODEL
        FINAL_DATA_MODEL --> S4_GEN_MD
        FINAL_DATA_MODEL --> S4_GEN_JSON
        S4_GEN_MD --> MD_OUTPUT
        S4_GEN_JSON --> JSON_OUTPUT
    end

    ASSESSMENT_REPORT --> S4_MERGE
    DIAGNOSTIC_REPORT --> S4_MERGE
    REFINED_PROMPT --> S4_MERGE
```

## Features

-   **Multi-Provider Support**: Seamlessly switch between **Google Gemini** (via Vertex/AI Studio) and **OpenAI** (GPT-4o, o1, o3) models.
-   **Hybrid LLM Routing**: Assign different models to specific pipeline steps (e.g., GPT-4o for extraction, Gemini 3.1 Pro for synthesis) within a single execution profile.
-   **Multi-Stage Execution**: Modular workflow covering preprocessing, assessment, diagnostics, and results generation.
-   **Deterministic Caching**: Content-based hashing ensures consistent artifact management and avoids redundant computation at both the application and API levels.
-   **SQLite Artifact Store**: Local persistence layer for caching LLM responses, reducing latency and costs during development and large-scale runs.
-   **Automated Diagnostic Reporting**: Automatically identifies discrepancies between model predictions and ground truth, performing root-cause analysis on mismatches.
-   **Granular Execution**: Run the full pipeline, specific stages, or individual atomic steps via the simplified CLI.

## Project Structure

-   `src/research_evaluation_pipeline`: Core logic, orchestrators, and CLI implementation.
    -   `clients/`: LLM provider implementations (Gemini, OpenAI) and factory routing.
    -   `logic/`: Stage-specific logic for assessment, diagnostics, and preprocessing.
    -   `core/`: Fundamental models, enums, and the `ArtifactStore`.
    -   `service/`: Internal services for prompt management and paper context handling.
-   `resources/`: Directory for input data and configurations.
-   `resources/profiles`: Execution strategy profiles (TOML).
-   `resources/papers`: Target directory for source PDF documents.
-   `resources/prompts_master.yaml`: The primary registry for evaluation criteria.
-   `output/`: Destination for generated JSON and Markdown reports.

## Installation

### Prerequisites

This project requires `uv` for dependency management and Python execution.

#### Installing uv via Homebrew (Recommended)

To install `uv` on macOS using Homebrew, run:

```bash
brew install uv
```

#### Installing uv via Curl

Alternatively, you can install `uv` using the official installation script:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Project Setup

**Synchronize Environment**: Navigate to the project directory and run:
```bash
uv sync
```

## Data Structures

### Prompt Registry
The pipeline supports various formats for evaluation criteria:

-   **Standalone Files** (`.md`, `.txt`): The entire file content is treated as the master prompt.
-   **Prompt Registries** (`.yaml`, `.json`): A dictionary of multiple prompts. Requires the `--prompt-key` argument to select a specific instruction set from the file.

### Ground Truth Data
To enable accuracy reporting and diagnostic analysis, your ground truth CSV must adhere to the following specification:

-   **Delimiter**: Semicolon (`;`).
-   **Required Columns**:
    -   `study_number`: Identifier matching the paper filename stem (e.g., `0400.pdf` -> `400`). Leading zeros are automatically stripped for matching.
    -   `prompt_number`: Criterion identifier matching the prompt registry (e.g., `1`, `2a`).
    -   `answer`: Binary integer where `1` is **True** and `0` is **False**.

## Configuration

### API Credentials

The pipeline utilizes the `keyring` library to securely manage API keys. Users must ensure that the appropriate API keys are stored in the system keychain and the correct service and account names are provided in `src/research_evaluation_pipeline/clients/factory.py`.

### Execution Profile Reference

Example execution profiles are defined in `resources/profiles/execution.toml`. Profiles control model selection, temperatures, and strategies for every step of the pipeline.

#### 1. General Step Settings
All atomic steps (e.g., Refinement, Decomposition, Synthesis) share a common foundation defined by the `StepSettings` model:

-   `model`: The identifier of the LLM model to use.
    -   *Options (Example Implementation)*: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-flash-preview`, `gemini-3.1-pro-preview`, etc.
-   `temperature`: Controls the randomness of the generation.
-   `cache_policy`: Rules for interacting with the local artifact store.
    -   `use-cache`: Use existing artifacts if available (default).
    -   `bypass-cache`: Execute the step and store the result, ignoring existing artifacts.
    -   `overwrite-cache`: Execute the step and overwrite any existing artifacts.

#### 2. Pipeline Configuration
Global settings that affect the overall data flow:

-   `ingestion_mode`: How paper content is provided to the models.
    -   `extraction`: Converts PDF to Markdown via a dedicated extraction step.
    -   `upload`: Directly uploads the PDF binary to the model's file API.

---

#### Stage 1: Preprocess
Prepares evaluation criteria and extracts paper content for processing.

-   **Refinement**: Structures and cleans the master criteria into a refined prompt.
    -   `strategy`:
        -   `standard`: Directly structures the master criteria.
        -   `semantic`: Resolves semantic ambiguities and improves criteria phrasing.
-   **Extraction**: Converts source PDFs into structured Markdown (active only if `ingestion_mode` is `extraction`).

---

#### Stage 2: Assessment
Assesses papers against defined criteria using single-pass or multi-step reasoning.

-   `fragmentation`: High-level execution architecture.
    -   `fast`: Performs a single-shot assessment.
    -   `plan`: Executes a multi-step chain.
-   **Decomposition**: Breaks down the evaluation criteria into atomic task groups.
    -   `strategy`:
        -   `semantic`: Groups criteria by logical intent.
        -   `structural`: Groups criteria based on the paper's section hierarchy.
-   **Extraction**: Locates and extracts verbatim evidence for each task.
    -   `strategy`:
        -   `standard`: Standard evidence location logic.
    -   `processing_mode`:
        -   `concurrent`: Parallel execution of evidence extraction tasks.
        -   `sequential`: Serial execution to manage API rate limits.
-   **Synthesis**: Renders final scores and reasoning based on extracted evidence.
    -   `strategy`:
        -   `analytical`: Balanced reasoning with evidence cross-referencing.
        -   `concise`: Final scores with minimal explanation.
        -   `verbose`: Comprehensive reports with exhaustive evidence logs.

---

#### Stage 3: Diagnostic (Optional)
Identifies and analyzes discrepancies between model assessments and ground truth.

-   `fragmentation`:
    -   `fast`: Single-shot diagnostic analysis.
    -   `plan`: Multi-step diagnostic analysis.
-   `prompt_source`: Determines which criteria prompt to use for analysis.
    -   `master`: Uses original evaluation criteria.
    -   `refined`: Uses the prompt from the Preprocess stage.
-   **Decomposition**: Groups assessment errors for thematic analysis.
    -   `strategy`:
        -   `thematic`: Groups errors by common logical themes.
-   **Analysis**: Performs the root-cause analysis on the identified errors.
    -   `strategy`:
        -   `diagnose-all`: Analyzes every prediction.
        -   `diagnose-mismatches`: Analyzes only model/ground-truth disagreements.
        -   `diagnose-matches`: Analyzes cases where the model correctly matched ground truth.
        -   `diagnose-overpredictions`: Specifically targets false positives.
        -   `diagnose-underpredictions`: Specifically targets false negatives.
    -   `processing_mode`:
        -   `concurrent`: Parallel analysis.
        -   `sequential`: Serial analysis.

---

#### Stage 4: Results
Aggregates results and generates final reports.

-   **Artifact Reconstruction**: Merges fragmented assessment and diagnostic artifacts into a unified data model.
-   **Multi-Format Export**: Generates human-readable Markdown and machine-readable JSON reports.
-   **Ground Truth Comparison**: Calculates accuracy metrics and identifies discrepancies between model predictions and ground truth.

### Provider Architecture

The pipeline uses a multi-client architecture to support various LLM providers. Google Gemini is the current reference implementation.

## Usage

The primary entry point for the pipeline is the `rrp` command.

### Run Full Pipeline

Execute the end-to-end assessment and diagnostic for a specific paper:

```bash
uv run rrp run-pipeline \
    --profile <profile_name> \
    --paper-path <path_to_pdf> \
    --prompt-path <path_to_prompt_file> \
    --prompt-key <optional_key> \
    --ground-truth-path <path_to_csv>
```

### Run Specific Stage

Execute a single stage (preprocess, assessment, diagnostic, result):

```bash
uv run rrp run-stage <stage_name> \
    --paper-path <paper_path> \
    --prompt-path <prompt_path> \
    --prompt-key <prompt_key> \
    --ground-truth-path <ground_truth_path> \
    --profile <profile_name> \
    --client-profile <client_profile_name>
```

### Run Specific Step

Execute a granular atomic step (e.g., refine, extract, decompose, synthesize, analyze):

```bash
uv run rrp run-step <stage_name> <step_name> \
    --paper-path <paper_path> \
    --prompt-path <prompt_path> \
    --prompt-key <prompt_key> \
    --ground-truth-path <ground_truth_path> \
    --profile <profile_name> \
    --client-profile <client_profile_name>
```

### Database Management

The `db` command group provides utilities for managing the local artifact cache.

-   **Show status**: `uv run rrp db status` (Displays entry counts and stats)
-   **Wipe database**: `uv run rrp db clear`
-   **Clear stage**: `uv run rrp db clear-stage <stage_name>`
-   **Seed database**: `uv run rrp db seed` (Restores convenience artifacts)
-   **Capture artifacts**: `uv run rrp db capture` (Exports DB to convenience files)

### Convenience Scripts

The `scripts/` directory contains shell scripts that simplify common execution patterns by leveraging the `resources/` directory conventions.

#### Script Configuration
These scripts are pre-configured to map the following local paths to the corresponding `rrp` CLI arguments:
-   `--paper-path`: Maps to `resources/papers/<paper_name>.pdf`
-   `--prompt-path`: Maps to `resources/prompts_master.yaml`
-   `--ground-truth-path`: Maps to `resources/correct_answers.csv`

#### Populating Resources for Scripts
To use these scripts, organize your data as follows:
1.  **Papers**: Add source PDFs to `resources/papers/`.
2.  **Ground Truth**: Name your CSV `correct_answers.csv` and place it in the `resources/` directory.
3.  **Criteria**: Populate `resources/prompts_master.yaml` with your criteria (this file includes several example prompts for reference).

**Available Scripts**:
-   `run_pipeline.sh`: Executes the full end-to-end pipeline.
-   `run_stage.sh`: Executes a specific pipeline stage (e.g., `preprocess`).
-   `run_step.sh`: Executes a granular atomic step (e.g., `refine`).

**Example Usage**:
```bash
./scripts/run_pipeline.sh <profile_name> <paper_path> <prompt_path> <prompt_key> <ground_truth_path>
```
*Note: Scripts default to values in `resources/` if arguments are omitted.*

## Development

### Logging & Observability

The pipeline uses `loguru` for high-signal logging. Every model interaction is logged with the specific model name and token usage (where available) to provide clear visibility into the hybrid execution flow.

### Running Tests

Execute the test suite using `pytest`:

```bash
uv run pytest
```
