# AI-Assisted RRP Analysis Tool

## Overview

The **AI-Assisted RRP (Responsible Research Practices) Analysis Tool** is an advanced, agentic evaluation pipeline built to assess research papers against predefined scientific and methodological criteria. The system reads research paper PDFs, converts them to structured markdown using multimodal extraction, and leverages Google's Gemini models to assess the presence or absence of Responsible Research Practices (RRPs).

The tool is highly modular, configurable via a CLI (`main.py`), and designed to enforce strict structural constraints using Pydantic schemas. It supports both a "Fast" single-pass evaluation and a complex, multi-stage "Planning" evaluation that mimics human cognitive processes to mitigate AI hallucinations and affirmative bias.

## Core Architecture

- **`main.py`**: The CLI entry point. It orchestrates the processing of single PDFs or entire directories, sets up logging, initiates the evaluation modes, and invokes the comparison and ground-truth validation routines.
- **`agent.py`**: The core LLM orchestration layer. It handles interactions with the Google GenAI API, providing functions for PDF multimodal extraction (`process_pdf`), dynamic prompt sanitization (`refine_prompt`), and the execution logic for the evaluation modes (`run_fast_mode` and `run_planning_mode`).
- **`evaluate.py`**: The scoring and comparison engine. It compares the model's generated JSON assessment against a ground truth CSV (loaded via `load_expected_answers`), calculating accuracy metrics. It also features a routine to generate diff reports between two JSON artifacts.
- **`schema.py`**: Defines the rigorous Pydantic data models enforcing structured JSON outputs from the LLM, including `TaskListArtifact`, `EvidenceArtifact`, `AssessmentReportArtifact`, and `EvaluationMetrics`.
- **`config.py`**: Manages environment variables and securely retrieves API keys from the local keychain (via `keyring`), abstracting configuration from the application logic.
- **`logger.py`**: Implements an aesthetically pleasing, customized CLI logger using `rich`, muting noisy underlying API logs to provide clean, semantic "cognitive step" readouts to the user.

## Execution Pipelines

The tool routes the parsed markdown and the master criteria through one of two distinct pipelines:

### 1. Fast Mode
A rapid, single-shot evaluation. The entire markdown text and the cleaned master criteria are fed into the reasoning model in a single prompt. The model outputs a definitive `FastAssessmentResult` containing boolean answers and justifications. This is fast but more susceptible to hallucinations on complex or highly nuanced papers.

### 2. Planning Mode
A multi-stage, methodical evaluation pipeline engineered for high fidelity and traceability.
1. **Task Generation**: The master criteria is broken down logically into `TaskGroup`s and specific `SubTask`s.
2. **Evidence Extraction**: The model acts as a focused extractor. It scans the paper specifically to locate and copy exact quotes addressing the tasks. This step can be executed iteratively per question (`isolated` strategy) or per group (`batch` strategy).
3. **Synthesis & Assessment**: A final reasoning pass evaluates the extracted evidence (ignoring the original document to strictly prevent affirmative bias) to render final True/False decisions and justifications, outputting an `AssessmentReportArtifact`.

## Evaluation & Metrics

The system natively supports evaluating its own performance. By providing an expected CSV file mapping study numbers and prompt numbers to correct answers, the tool calculates accuracy percentages and flags diverging answers, storing all artifacts in an organized timestamped `output/` directory.

## Pipeline Flow Visualization

```mermaid
flowchart TD
    %% Input Sources
    Doc[(PDF Document)] --> Extractor
    Prompt[(Master Prompt)] --> Refiner
    
    %% Pre-Processing
    Extractor(process_pdf\nMultimodal Extraction) --> MD[Markdown Document]
    Refiner(refine_prompt\nLLM Prompt Cleaning) --> CleanPrompt[Cleaned Master Criteria]
    
    MD --> Router{Execution Mode}
    CleanPrompt --> Router
    
    %% --- FAST MODE ---
    Router -->|--mode fast| FastRun(Reasoning Model:\nSingle-Shot Evaluation)
    FastRun --> FastResult[FastAssessmentResult\nJSON Artifact]
    
    %% --- PLANNING MODE ---
    Router -->|--mode planning| TaskGen(Reasoning Model:\nTask Generation)
    TaskGen --> TaskList[TaskListArtifact]
    
    TaskList --> EvidExtr(Reasoning Model:\nEvidence Extraction\nIsolated or Batch)
    MD --> EvidExtr
    EvidExtr --> Evidence[EvidenceArtifacts\nExact Quotes]
    
    Evidence --> FinalSynth(Reasoning Model:\nFinal Synthesis)
    CleanPrompt --> FinalSynth
    FinalSynth --> PlanResult[AssessmentReport\nJSON Artifact]
    
    %% Evaluation Phase
    FastResult --> EvalEngine(evaluate.py\ncompare_results)
    PlanResult --> EvalEngine
    GT[(Ground Truth CSV)] --> EvalEngine
    
    EvalEngine --> FinalMetrics([EvaluationMetrics\nAccuracy & Diff Reports])
    
    classDef file fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bbf,stroke:#333,stroke-width:2px;
    classDef artifact fill:#dfd,stroke:#333,stroke-width:2px;
    classDef decision fill:#ffd,stroke:#333,stroke-width:2px;
    
    class MD,CleanPrompt,FastResult,TaskList,Evidence,PlanResult,FinalMetrics artifact;
    class Extractor,Refiner,FastRun,TaskGen,EvidExtr,FinalSynth,EvalEngine process;
    class Doc,Prompt,GT file;
    class Router decision;
```
