# Research Evaluation Pipeline

The Research Evaluation Pipeline is a framework designed for the automated assessment and diagnostic analysis of research papers using large language models. It provides a structured workflow to ingest documents, evaluate them against specific criteria, and perform root-cause analysis on discrepancies between model predictions and ground truth data.

## Execution Architecture

The following diagram illustrates the complete data flow and execution architecture of the pipeline. It highlights the production of granular artifacts within each stage and the cross-stage dependencies that drive the reasoning process.

```mermaid
graph TD
    %% Global Inputs
    PDF_FILE["Source PDF File"]
    MASTER_PROMPT["Master Prompt (YAML/MD)"]
    GROUND_TRUTH["Ground Truth (CSV)"]

    %% Stage 1: Preprocess
    subgraph PREPROCESS["Stage 1: Preprocess"]
        direction TB
        S1_REFINE["Step: Refine Prompt"]
        S1_EXTRACT["Step: Extraction"]
        S1_UPLOAD["Step: API Upload"]
        REFINED_PROMPT["ARTIFACT: Refined Prompt"]
        PAPER_CONTEXT["ARTIFACT: Paper Context (Markdown or API Ref)"]

        S1_REFINE --> REFINED_PROMPT
        S1_EXTRACT --> PAPER_CONTEXT
        S1_UPLOAD --> PAPER_CONTEXT
    end

    PDF_FILE --> S1_EXTRACT
    PDF_FILE --> S1_UPLOAD
    MASTER_PROMPT --> S1_REFINE

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

- **Multi-Stage Pipeline**: Modular execution flow including preprocessing, assessment, diagnostic, and results stages.
- **Automated Reporting**: Generation of comprehensive Markdown and JSON reports comparing model assessments against ground truth data.
- **Granular Control**: Ability to run the entire pipeline, specific stages, or individual atomic steps.
- **Deterministic Tracking**: Content-based hashing and key building for consistent artifact management and caching.
- **Multi-Client Provider Architecture**: Protocol-based architecture designed to support multiple LLM providers. A Google Gemini client is currently provided as the reference implementation.
- **Flexible Configuration**: Support for multiple execution profiles and client configurations via TOML files.
- **Persistent Artifact Store**: SQLite-based caching system to minimize redundant API calls and facilitate development.

## Project Structure

- `src/research_evaluation_pipeline`: Core logic and CLI implementation.
- `resources/`: Directory for input data and configurations. **Note**: As specified in `.gitignore`, local assets such as PDFs, databases, and convenience artifacts are excluded from the repository and must be populated by the user before execution.
- `resources/profiles`: Example configuration files for execution strategies and client settings.
- `resources/papers`: Target directory for source PDF documents.
- `resources/prompts_default.yaml`: A provided set of default system and user prompts for various pipeline stages.
- `resources/prompts_master.yaml`: The primary registry for evaluation criteria. This file should be populated by the user with specific master instructions.
- `output`: Destination for generated JSON and Markdown reports.

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
**Populate Resources**:
-   Add your research papers (PDFs) to the `resources/papers/` directory.
-   Ensure your ground truth files are available in the `resources/` folder.
-   While a default set of pipeline prompts is provided in `resources/prompts_default.yaml`, you should populate `resources/prompts_master.yaml` with your specific evaluation criteria.

## Configuration

### API Credentials

The pipeline utilizes the `keyring` library to securely manage API keys. Users must ensure that the appropriate API keys are stored in the system keychain using the service and account identifiers defined in the client profiles (e.g., `resources/profiles/client.toml`).

### Execution Profile Reference

Example execution profiles are defined in `resources/profiles/execution.toml`. Each profile controls the behavior, model selection, and strategies for the entire pipeline.

#### Global Settings
- `ingestion_mode`: Determines how paper content is provided to models (`extraction` for Markdown conversion, `upload` for direct PDF binary upload).

#### Preprocess Stage
- **Refinement**: Configures how master criteria are cleaned. Includes `model`, `temperature`, `cache_policy`, and `strategy` (`standard` or `semantic`).
- **Extraction**: Configures PDF-to-Markdown conversion. Includes `model`, `temperature`, and `cache_policy`.

#### Assessment Stage
- `fragmentation`: High-level execution mode. `fast` for single-pass assessment, `plan` for multi-step reasoning (decomposition -> extraction -> synthesis).
- **Decomposition**: Settings for breaking down criteria. Strategy can be `semantic` or `structural`.
- **Extraction**: Settings for evidence location. Includes `processing_mode` (`sequential` or `concurrent`).
- **Synthesis**: Settings for final reasoning over evidence. Strategy can be `concise`, `analytical`, or `verbose`.

#### Diagnostic Stage (Optional)
- `fragmentation`: `fast` for single-pass analysis, `plan` for multi-step root cause analysis.
- `prompt_source`: Determines whether to use the `master` or `refined` prompt for diagnostics.
- **Decomposition**: Settings for error batching. Strategy defaults to `thematic`.
- **Analysis**: Settings for mismatch detection. Strategies include `diagnose-all`, `diagnose-mismatches`, `diagnose-matches`, etc.

#### Results Stage (Optional)
- **Artifact Reconstruction**: Logic for merging fragmented assessment and diagnostic artifacts into a unified data model.
- **Multi-Format Export**: Automated generation of results in both human-readable Markdown and machine-readable JSON formats.
- **Ground Truth Comparison**: Integrated logic for calculating accuracy and identifying discrepancies between model predictions and provided ground truth.

### Provider Architecture

The system is designed with a multi-client provider architecture to support various LLM services. The Google Gemini client is currently implemented as the primary example of this architecture.

## Usage

The primary entry point for the pipeline is the `rrp` command.

### Run Full Pipeline

Execute the end-to-end assessment for a specific paper:

```bash
uv run rrp run-pipeline \
    --paper-path <paper_path> \
    --prompt-path <prompt_path> \
    --prompt-key <prompt_key> \
    --ground-truth-path <ground_truth_path> \
    --profile <profile_name> \
    --client-profile <client_profile_name> \
    --execution-profiles <execution_profiles_path> \
    --client-profiles <client_profiles_path>
```

### Run Specific Stage

Execute a single stage of the research pipeline (preprocess, assessment, diagnostic, results):

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

- **Wipe database**: `uv run rrp db clear`
- **Seed database**: `uv run rrp db seed`
- **Capture artifacts**: `uv run rrp db capture`

### Convenience Scripts

The `scripts/` directory contains shell scripts that provide shortcuts for common execution patterns. These scripts are configured to use the suggested directory structure within the `resources/` folder to streamline the workflow.

- `run_pipeline.sh`: Runs the full pipeline with configurable parameters.
- `run_stage.sh`: Runs a specific pipeline stage.
- `run_step.sh`: Runs a granular atomic step.

Example:
```bash
./scripts/run_pipeline.sh <profile_name> <client_profile_name> <paper_path>
```

## Development

### Running Tests

Execute the test suite using `pytest`:

```bash
uv run pytest
```
