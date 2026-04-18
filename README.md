# Hyper Simulation Project

This repository implements and evaluates a question-answering consistency analysis framework based on hypergraph representation and hypergraph simulation.
This README is written for paper code submission: it explains what each part of the codebase is responsible for.

## 1. What This Repository Does

The core idea is to convert questions and candidate documents into hypergraph structures, then use structural matching (hyper simulation) to determine whether evidence supports the question semantics. The repository also includes multiple baselines and multi-dataset evaluation pipelines.

The project has three layers:

1. Python main pipeline (data loading, graph construction, inference, evaluation)
2. Rust simulation backend (high-performance graph/hypergraph simulation algorithms)
3. Task scripts and utilities (dataset building, batch runs, statistics, debugging)

## 2. Top-Level Directory Guide

- src/hyper_simulation/: main method implementation and experiment logic (core package)
- src/contrievers/: retrieval model and indexing modules (Contriever-based)
- src/simulation/: Python bindings for Rust simulation algorithms
- src/graph-simulation/: Rust graph simulation library
- tests/: experiment entry scripts, task scripts, debugging scripts
- scripts/: shell wrappers for frequent batch operations
- pixi.toml: environment, dependencies, and task command definitions
- pyproject.toml: Python package metadata
- remove_comment.py: auxiliary text-processing script

## 3. Core Code Areas

### 3.1 src/hyper_simulation/

This is the primary implementation of the paper method.

#### A. Data Instance and Task Adaptation

- query_instance.py
  - Defines the QueryInstance structure (query, candidate documents, answers, logs, etc.).
  - build_query_instance_for_task converts samples from different datasets (HotpotQA, MuSiQue, MultiHop, LegalBench, ARC, etc.) into a unified internal format.

#### B. QA Main Pipelines

- question_answer/rag_no_retrival.py
  - One of the main experiment entry points.
  - Handles data loading, method selection (vanilla, hyper_simulation, and baselines), prompt building, model calls, evaluation, and incremental saving.
  - Supports resume-from-checkpoint, prompt-only export, and batch-wise persistence.

- question_answer/rag.py
  - Retrieval-augmented QA pipeline.
  - Includes Contriever retrieval, index loading/building, evidence assembly, and final answer generation.

- question_answer/decompose.py
  - Multi-hop question decomposition and sub-question to vertex alignment.
  - Uses LLM output in structured JSON form (sub-questions with aligned vertex ids).

#### C. Method Components (Algorithm Core)

- component/build_hypergraph.py
  - Text -> spaCy/coreference parsing -> dependency processing -> hypergraph construction.
  - Supports single and batched (including GPU pipeline) graph building with on-disk cache artifacts (query/data pkl files).

- component/consistent.py
  - Consistency detection and query fixup entry.
  - Loads constructed hypergraphs, performs distance-based filtering plus hyper-simulation coverage checking.
  - Uses fusion for multi-hop tasks; outputs support/inconsistency signals and evidence notes for single-hop tasks.

- component/hyper_simulation.py
  - Core algorithmic process.
  - Converts local hypergraph structures to simulation backend structures, filters candidate matches, builds semantic clusters, computes D-Match, and generates final simulation mappings.

- Other modules in this folder (denial.py, d_match.py, semantic_cluster.py, embedding.py, nli.py, postprocess.py)
  - Handle candidate filtering, match scoring, semantic-cluster processing, embedding similarity, NLI-related logic, and result post-processing.

#### D. Hypergraph Structure and Linguistic Processing

- hypergraph/
  - Defines vertices, hyperedges, relations, and dependency abstractions.
  - Handles entities, POS tags, coreference, abstraction mapping, and graph fusion.
  - hypergraph/hypergraph.py defines the core data structures.

#### E. Baseline Methods

- baselines/
  - Includes contradoc, sparseCL, sentLI, CDIT, and BSIM baselines.
  - Switched uniformly via --method in evaluation scripts.

#### F. LLM and General Utilities

- llm/
  - Wrappers for chat completion, text completion, timing statistics, and prompt templates.

- utils/
  - Common utilities such as logging and text cleaning.

### 3.2 src/contrievers/

Retrieval stack implementation, including:

- Data processing and normalization
- Vector indexing and retrieval
- Distributed/training helpers
- Retrieval evaluation and finetuning data construction

This part mainly supports the retrieval stage in question_answer/rag.py.

### 3.3 src/simulation/ (Python Binding Layer)

Python exposure layer for Rust crates (pyo3 + maturin):

- Provides graph/hypergraph simulation interfaces to Python
- Serves as the low-level compute dependency for hyper_simulation components

### 3.4 src/graph-simulation/ (Rust Algorithm Layer)

Pure Rust graph simulation library:

- Core simulation/isomorphism-related algorithms
- Benchmarks and tests
- Indirectly used by src/simulation/

## 4. Experiment and Script Directories

### 4.1 tests/tasks/

Centralized experiment scripts for paper evaluation, organized by dataset and purpose:

- build_*.py: build/prepare dataset samples
- *_baseline.py: run baseline variants
- *.py (e.g., musique.py, hotpotqa.py): run main method experiments
- *_time.py: runtime statistics
- split_chunk.py, count_tokens.py: chunking and token-length statistics

### 4.2 tests/tools/

Utility scripts for development and diagnosis:

- Hypergraph build/debug and inspection
- WordNet/spaCy debugging
- Data cleanup and title processing

### 4.3 tests/test_*.py

Lightweight tests and smoke tests:

- Core pipeline availability checks
- Quick regression checks for selected tasks

## 5. Configuration Files

- pixi.toml
  - Defines multi-environment dependencies and task commands (naive/hypergraph/simulation features).
  - Includes commonly used experiment command mappings.

- pyproject.toml
  - Python project metadata and build configuration.

## 6. Typical End-to-End Flow (Reading Perspective)

From a code-reading (not execution-first) perspective, the main pipeline is:

1. Task scripts load samples and convert them into QueryInstance objects.
2. A method branch is selected: direct QA, baseline, or hyper_simulation.
3. If hyper_simulation is selected: text-to-hypergraph construction -> hyper simulation matching -> consistency-aware context marking.
4. Prompt assembly and LLM answer generation.
5. Post-processing, metric computation, and JSON result persistence.

## 7. Suggested Reading Order

If you only want to understand the paper method quickly, read in this order:

1. src/hyper_simulation/question_answer/rag_no_retrival.py
2. src/hyper_simulation/component/consistent.py
3. src/hyper_simulation/component/build_hypergraph.py
4. src/hyper_simulation/component/hyper_simulation.py
5. src/hyper_simulation/hypergraph/hypergraph.py
6. tests/tasks/*.py (dataset-specific experiment entries)
