A# Phase 3A Performance Analysis - Implementation Plan

**PDCA Phase**: PLAN - Detailed Planning
**Date**: 2025-12-08
**Project**: LangSmith Trace Data Export & Analysis
**Prerequisites**: See `phase3a_performance_analysis.md` for analysis findings

---

## Document Purpose

This document provides the detailed implementation plan for Phase 3A Performance Analysis tool, following PDCA methodology and TDD practices.

**Related Documents**:
- Analysis: `phase3a_performance_analysis.md`
- Requirements: Requirements specification document

---

## Implementation Strategy

### Core Data Structures

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

@dataclass
class Trace:
    """Single LangSmith trace/run."""
    id: str
    name: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: float
    status: str
    run_type: str  # 'llm', 'chain', 'tool'
    parent_id: Optional[str]
    child_ids: List[str]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    error: Optional[str]

    @property
    def is_workflow(self) -> bool:
        """Check if this is a workflow root (LangGraph)."""
        return self.name == "LangGraph" and self.run_type == "chain"

    @property
    def is_node(self) -> bool:
        """Check if this is a workflow node."""
        return self.run_type == "chain" and not self.is_workflow

@dataclass
class Workflow:
    """Complete workflow execution (LangGraph + children)."""
    root_trace: Trace
    nodes: Dict[str, List[Trace]]  # node_name -> executions
    all_traces: List[Trace]

    @property
    def total_duration(self) -> float:
        return self.root_trace.duration_seconds

    def get_node_durations(self) -> Dict[str, float]:
        """Calculate average duration per node type."""
        ...

    def get_parallel_execution_evidence(self) -> Dict[str, Any]:
        """Analyze validator start times for parallel execution."""
        ...

@dataclass
class TraceDataset:
    """Container for all loaded trace data."""
    workflows: List[Workflow]
    orphan_traces: List[Trace]  # Traces without parent (flat data)
    metadata: Dict[str, Any]
    is_hierarchical: bool  # True if child relationships present

    def get_workflow_durations(self) -> List[float]:
        """Extract all workflow durations for percentile analysis."""
        return [w.total_duration for w in self.workflows if w.total_duration > 0]
```

### Dual Data Loading

**Mode 1: Load from JSON Export (Flat Data)**
```python
def load_from_json(filepath: str) -> TraceDataset:
    """
    Load trace data from JSON export file.

    Limitation: May only have top-level traces if child_runs not populated.
    Provides graceful degradation for analysis.

    Args:
        filepath: Path to JSON export file

    Returns:
        TraceDataset with loaded traces

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON format invalid
    """
    # Implementation:
    # 1. Read JSON file
    # 2. Parse export_metadata and traces
    # 3. Convert each trace dict to Trace object
    # 4. Build trace index by ID
    # 5. Attempt to reconstruct hierarchies from parent_id if present
    # 6. Group into Workflow objects
    # 7. Validate data quality
    # 8. Return TraceDataset
```

**Mode 2: Fetch from Live API (Hierarchical Data)**
```python
def load_from_langsmith_api(
    api_key: str,
    project_name: str,
    limit: int,
    trace_ids: Optional[List[str]] = None
) -> TraceDataset:
    """
    Fetch trace data from LangSmith API with hierarchical relationships.

    Args:
        api_key: LangSmith API key
        project_name: Project name or ID
        limit: Maximum number of workflows to fetch
        trace_ids: Optional list of specific trace IDs to fetch

    Returns:
        TraceDataset with hierarchical data (is_hierarchical=True)

    Raises:
        AuthenticationError: If API key invalid
        ProjectNotFoundError: If project doesn't exist

    Implementation:
    # 1. Initialize LangSmith client (reuse from export_langsmith_traces.py)
    # 2. Fetch root traces (LangGraph workflows)
    # 3. For each workflow:
    #    a. Fetch full run tree with children
    #    b. Build Workflow object with all related traces
    # 4. Validate data quality
    # 5. Return TraceDataset with is_hierarchical=True

    # Research needed: Determine SDK method for hierarchical fetching:
    # - client.read_run(run_id)?
    # - client.get_run_tree(run_id)?
    # - Recursive fetching via parent_run_id field?
    """
```

### Analysis Functions

#### 1. Latency Distribution Analysis

```python
@dataclass
class LatencyDistribution:
    """Results of latency distribution analysis."""
    p50_minutes: float
    p95_minutes: float
    p99_minutes: float
    min_minutes: float
    max_minutes: float
    mean_minutes: float
    std_dev_minutes: float
    sample_size: int
    outliers_above_23min: List[str]  # trace IDs
    outliers_below_7min: List[str]   # trace IDs
    percent_within_7_23_claim: float

    def to_csv(self, filepath: str) -> None:
        """Export results to CSV file."""
        ...

def analyze_latency_distribution(
    workflows: List[Workflow]
) -> LatencyDistribution:
    """
    Calculate latency percentiles and validate timing claims.

    Algorithm:
    1. Extract duration_seconds from each workflow.root_trace
    2. Convert to minutes
    3. Filter out incomplete workflows (duration = 0)
    4. Sort durations
    5. Calculate percentiles using numpy.percentile()
    6. Identify outliers outside 7-23 min range
    7. Calculate % within claimed range

    Edge Cases:
    - Require minimum 10 workflows for statistical validity
    - Handle None durations gracefully
    - Warn if sample size too small
    """
```

#### 2. Bottleneck Identification

```python
@dataclass
class NodePerformance:
    """Performance metrics for a single node type."""
    node_name: str
    execution_count: int
    avg_duration_seconds: float
    median_duration_seconds: float
    std_dev_seconds: float
    min_duration_seconds: float
    max_duration_seconds: float
    avg_percent_of_workflow: float
    total_time_across_workflows: float

@dataclass
class BottleneckAnalysis:
    """Results of bottleneck identification."""
    node_performances: List[NodePerformance]
    primary_bottleneck: str  # Node name consuming most time
    top_3_bottlenecks: List[str]
    total_workflows_analyzed: int

    def to_csv(self, filepath: str) -> None:
        """Export results to CSV file."""
        ...

def identify_bottlenecks(
    workflows: List[Workflow]
) -> BottleneckAnalysis:
    """
    Analyze which nodes consume the most time.

    Algorithm:
    1. For each workflow:
       a. Extract all child nodes
       b. Calculate each node's duration
       c. Calculate each node's % of workflow total
    2. Aggregate across all workflows:
       a. Group by node name
       b. Calculate mean, median, std dev per node type
       c. Calculate avg % of workflow time
    3. Rank nodes by avg duration or % of workflow
    4. Identify top 3 bottlenecks

    Priority Nodes:
    - generate_spec, xml_transformation
    - meta_evaluation_and_validation, normative_validation, simulated_testing
    - fix_xml_node, import_step

    Edge Cases:
    - Handle missing child_runs (flat data): Return warning + limited analysis
    - Handle nodes that don't appear in all workflows (e.g., fix_xml_node only on errors)
    - Skip LLM calls (too granular), focus on chain-level nodes
    """
```

#### 3. Parallel Execution Verification

```python
@dataclass
class ParallelExecutionEvidence:
    """Evidence for or against parallel execution."""
    validator_names: List[str]
    workflows_analyzed: int
    parallel_confirmed_count: int
    sequential_count: int

    avg_start_time_delta_seconds: float
    max_start_time_delta_seconds: float

    avg_sequential_time_seconds: float  # Sum of all 3 durations
    avg_parallel_time_seconds: float    # Max of 3 durations
    avg_time_savings_seconds: float     # Sequential - Parallel
    avg_time_savings_percent: float

    is_parallel: bool
    confidence: str  # 'high', 'medium', 'low'
    evidence_quality: str  # 'strong', 'weak', 'insufficient'

    def to_csv(self, filepath: str) -> None:
        """Export results to CSV file."""
        ...

def verify_parallel_execution(
    workflows: List[Workflow]
) -> ParallelExecutionEvidence:
    """
    Analyze validator execution patterns to confirm parallel execution.

    Algorithm:
    1. For each workflow:
       a. Find validator nodes (meta_evaluation, normative, simulated)
       b. Extract start_time for each validator
       c. Calculate time delta between starts:
          - If all start within ~5 seconds: PARALLEL
          - If starts are staggered by >1 minute: SEQUENTIAL
       d. Calculate actual parallel time (max duration)
       e. Calculate hypothetical sequential time (sum durations)
       f. Calculate time savings

    2. Aggregate across workflows:
       a. Count parallel vs sequential patterns
       b. Calculate avg time savings
       c. Determine confidence level

    Confidence Criteria:
    - HIGH: >80% of workflows show parallel pattern, avg start delta <10s
    - MEDIUM: 50-80% show parallel, avg start delta <30s
    - LOW: <50% show parallel or insufficient data

    Edge Cases:
    - Missing validator nodes: Flag as 'insufficient data'
    - Missing start_time: Cannot analyze, return 'insufficient data'
    - Only 1-2 validators present: Flag as 'unexpected workflow structure'
    - Flat data (no children): Return 'requires hierarchical data'
    """
```

---

## File Structure

```
export-langsmith-data/
├── plans/                         # NEW: Planning documents
│   ├── phase3a_performance_analysis.md
│   └── phase3a_performance_implementation_plan.md
├── analyze_traces.py              # NEW: Main analysis module (~688 lines)
│   ├── Data structures (Trace, Workflow, TraceDataset)
│   ├── Loading functions (load_from_json, load_from_langsmith_api)
│   ├── Analysis functions (3 main + helpers)
│   └── CSV export methods
├── test_analyze_traces.py         # NEW: Test suite (35+ tests)
│   ├── TestLoadFromJSON (5 tests)
│   ├── TestLoadFromAPI (2 tests)
│   ├── TestValidateTraceData (2 tests)
│   ├── TestLatencyDistribution (5 tests)
│   ├── TestBottleneckIdentification (4 tests)
│   ├── TestParallelExecutionVerification (5 tests)
│   ├── TestCSVExport (3 tests)
│   └── TestIntegration (2 tests)
├── notebooks/                     # NEW: Jupyter notebooks
│   └── phase3a_performance_analysis.ipynb
├── output/                        # NEW: Analysis outputs (gitignored)
│   ├── latency_distribution.csv
│   ├── bottleneck_analysis.csv
│   ├── parallel_execution_analysis.csv
│   ├── phase3a_performance_summary.md
│   └── *.png (visualizations)
├── fixtures/                      # NEW: Test data (scrubbed)
│   └── sample_traces_scrubbed.json
├── export_langsmith_traces.py     # EXISTING: Reuse for API client
├── test_export_langsmith_traces.py # EXISTING: No changes
├── requirements.txt               # MODIFY: Add analysis dependencies
├── .gitignore                     # MODIFY: Add output/ directory
└── README.md                      # MODIFY: Add analysis documentation
```

---

## Implementation Order (TDD)

### Phase 1: Data Loading Foundation (Day 1 Morning, 2-3 hours)

**RED - Write Failing Tests**:
```python
# test_analyze_traces.py
def test_load_valid_json_file():
    """Test loading a valid JSON export file."""
    dataset = load_from_json("fixtures/sample_traces_scrubbed.json")
    assert isinstance(dataset, TraceDataset)
    assert len(dataset.workflows) >= 0

def test_parse_datetime_formats():
    """Test parsing ISO 8601 timestamps."""
    dt = _parse_datetime("2025-12-01T10:00:00Z")
    assert dt is not None

def test_detect_flat_data():
    """Test detection of flat data (missing child_runs)."""
    dataset = load_from_json("fixtures/flat_data.json")
    assert dataset.is_hierarchical is False

def test_validate_missing_timestamps():
    """Test validation catches missing timestamps."""
    issues = validate_trace_data(dataset)
    timestamp_issues = [i for i in issues if "timestamp" in i.message]
    assert len(timestamp_issues) > 0
```

**GREEN - Implement Minimal Code**:
1. Create `Trace`, `Workflow`, `TraceDataset` dataclasses in `analyze_traces.py`
2. Implement `load_from_json()` function
3. Implement `_parse_datetime()` helper
4. Implement `validate_trace_data()` function

**REFACTOR - Clean Up**:
1. Extract common parsing logic into helpers
2. Add comprehensive type hints
3. Add detailed docstrings
4. Run Black, Ruff, mypy

### Phase 2: Latency Distribution Analysis (Day 1 Afternoon, 2-3 hours)

**RED - Write Failing Tests**:
```python
def test_calculate_percentiles():
    """Test percentile calculation with known values."""
    workflows = [_create_workflow(dur) for dur in [420, 600, 720, 900, 1200, 1380]]
    result = analyze_latency_distribution(workflows)
    assert 14 <= result.p50_minutes <= 16  # Should be ~15 min

def test_identify_outliers():
    """Test outlier detection."""
    workflows = [_create_workflow(dur) for dur in [360, 1200, 2400]]  # 6, 20, 40 min
    result = analyze_latency_distribution(workflows)
    assert len(result.outliers_below_7min) == 1
    assert len(result.outliers_above_23min) == 1

def test_percent_within_claim():
    """Test calculation of percent within 7-23 min claim."""
    durations = [420, 600, 720, 900, 1080, 1200, 1320, 2400]  # Last is 40 min
    workflows = [_create_workflow(dur) for dur in durations]
    result = analyze_latency_distribution(workflows)
    assert result.percent_within_7_23_claim == 87.5  # 7/8
```

**GREEN - Implement Analysis**:
1. Create `LatencyDistribution` dataclass
2. Implement `analyze_latency_distribution()` function
3. Use numpy.percentile() for calculations
4. Implement outlier detection logic

**REFACTOR - Optimize**:
1. Add error handling for edge cases (empty list, all zeros)
2. Optimize for large datasets
3. Add `to_csv()` method
4. Document algorithm

### Phase 3: Bottleneck Identification (Day 2 Morning, 3-4 hours)

**RED - Write Failing Tests**:
```python
def test_calculate_node_durations():
    """Test calculation of average node durations."""
    workflows = [sample_workflow] * 10  # 10 identical
    result = identify_bottlenecks(workflows)
    validator_perf = [n for n in result.node_performances
                      if n.node_name == "meta_evaluation_and_validation"][0]
    assert validator_perf.execution_count == 10
    assert validator_perf.avg_duration_seconds == 120.0

def test_identify_primary_bottleneck():
    """Test identification of slowest node."""
    workflow = _create_workflow_with_nodes({
        "generate_spec": 180,
        "xml_transformation": 500,  # Slowest
        "import_step": 100,
    })
    result = identify_bottlenecks([workflow])
    assert result.primary_bottleneck == "xml_transformation"

def test_handle_missing_child_runs():
    """Test handling of flat data (missing children)."""
    workflow = Workflow(root_trace=..., nodes={}, all_traces=[])
    result = identify_bottlenecks([workflow])
    # Should return warning or empty results
```

**GREEN - Implement Analysis**:
1. Create `NodePerformance`, `BottleneckAnalysis` dataclasses
2. Implement `identify_bottlenecks()` function
3. Group traces by node name
4. Calculate aggregates (mean, median, std dev)
5. Rank nodes by duration

**REFACTOR - Enhance**:
1. Add variability analysis (std dev)
2. Handle nodes that appear in subset of workflows
3. Add `to_csv()` method
4. Optimize for hierarchical vs flat data

### Phase 4: Parallel Execution Verification (Day 2 Afternoon, 2-3 hours)

**RED - Write Failing Tests**:
```python
def test_detect_parallel_validators():
    """Test detection of validators starting simultaneously."""
    # Create validators with start times within 5 seconds
    validators = [
        _create_trace("meta_evaluation", 120, start=base_time),
        _create_trace("normative_validation", 125, start=base_time + 2s),
        _create_trace("simulated_testing", 130, start=base_time + 3s),
    ]
    workflow = _create_workflow_with_traces(validators)
    result = verify_parallel_execution([workflow])
    assert result.is_parallel is True
    assert result.confidence == "high"

def test_calculate_time_savings():
    """Test calculation of time savings from parallelization."""
    # Validators: 120, 125, 130 seconds
    # Sequential: 375s, Parallel: 130s (max), Savings: 245s
    result = verify_parallel_execution([workflow])
    assert result.avg_sequential_time_seconds == 375
    assert result.avg_parallel_time_seconds == 130
    assert result.avg_time_savings_seconds == 245
```

**GREEN - Implement Detection**:
1. Create `ParallelExecutionEvidence` dataclass
2. Implement `verify_parallel_execution()` function
3. Extract validator traces by name
4. Calculate start time deltas
5. Compute time savings

**REFACTOR - Improve Confidence**:
1. Add confidence scoring algorithm
2. Handle edge cases (missing timestamps, partial sets)
3. Add `to_csv()` method
4. Document detection criteria

### Phase 5: Output Generation & Integration (Day 3 Morning, 2-3 hours)

**RED - Write Failing Tests**:
```python
def test_export_latency_csv(tmp_path):
    """Test exporting latency results to CSV."""
    result = LatencyDistribution(...)
    output_file = tmp_path / "latency.csv"
    result.to_csv(str(output_file))
    assert output_file.exists()
    # Verify CSV content

def test_full_analysis_workflow_from_json():
    """Test complete analysis pipeline from JSON file."""
    dataset = load_from_json("fixtures/sample.json")
    latency = analyze_latency_distribution(dataset.workflows)
    bottlenecks = identify_bottlenecks(dataset.workflows)
    parallel = verify_parallel_execution(dataset.workflows)
    # Verify all results
```

**GREEN - Implement Exports**:
1. Implement `to_csv()` methods for all result dataclasses
2. Implement `export_summary_markdown()` function
3. Create Jupyter notebook template
4. Add visualization helpers

**REFACTOR - Polish**:
1. Standardize CSV format across all exports
2. Add markdown table formatting
3. Ensure cross-platform paths
4. Add comprehensive docstrings

### Phase 6: API Loading & Hierarchical Data (Day 3 Afternoon, 3-4 hours) - OPTIONAL

**NOTE**: This phase is OPTIONAL if flat data analysis is sufficient initially.

**RED - Write Failing Tests**:
```python
@patch("analyze_traces.LangSmithExporter")
def test_load_from_api_success(mock_exporter_class):
    """Test successful API data fetch."""
    # Mock client and responses
    dataset = load_from_langsmith_api(...)
    assert isinstance(dataset, TraceDataset)
    assert dataset.is_hierarchical is True

@patch("analyze_traces.LangSmithExporter")
def test_load_from_api_auth_error(mock_exporter_class):
    """Test authentication error handling."""
    with pytest.raises(AuthenticationError):
        load_from_langsmith_api(api_key="invalid", ...)
```

**GREEN - Implement API Loading**:
1. Research LangSmith SDK methods for hierarchical fetching
2. Implement `load_from_langsmith_api()` function
3. Reuse `LangSmithExporter` client initialization
4. Fetch workflow trees with children

**REFACTOR - Optimize API Calls**:
1. Add rate limiting (reuse from export script)
2. Add progress indication for slow fetches
3. Handle partial failures gracefully

### Phase 7: Documentation & Testing (Day 4, 2-3 hours)

**Tasks**:
1. Update README.md with analysis module documentation
2. Complete Jupyter notebook with all sections and visualizations
3. Run full test suite: `pytest test_analyze_traces.py -v --cov=analyze_traces`
4. Run code quality checks: `black`, `ruff`, `mypy`, `bandit`
5. Generate coverage report (target >90%)
6. Test with actual sample data
7. Review and refine documentation

---

## Dependencies Update

**File**: `requirements.txt`

**Add**:
```python
# Analysis dependencies
numpy>=1.24.0      # Percentile calculations, array operations
pandas>=2.0.0      # Data manipulation, CSV I/O
matplotlib>=3.7.0  # Visualizations
seaborn>=0.12.0    # Statistical plots
jupyter>=1.0.0     # Notebook environment
ipykernel>=6.25.0  # Jupyter kernel
```

**Rationale**:
- NumPy: Efficient numerical operations, percentile calculations
- Pandas: DataFrame for data manipulation, easy CSV export
- Matplotlib/Seaborn: Visualizations in Jupyter notebooks
- Jupyter: Interactive analysis environment for end users

---

## Testing Strategy

### Test Organization

**File**: `test_analyze_traces.py` (35+ tests, ~800 lines)

**Test Classes**:
1. `TestLoadFromJSON` (5 tests) - File loading, schema validation, datetime parsing
2. `TestLoadFromAPI` (2 tests) - API fetch success/failure scenarios
3. `TestValidateTraceData` (2 tests) - Data quality checks
4. `TestLatencyDistribution` (5 tests) - Percentiles, outliers, filtering
5. `TestBottleneckIdentification` (4 tests) - Node aggregation, ranking, flat data handling
6. `TestParallelExecutionVerification` (5 tests) - Parallel detection, time savings, edge cases
7. `TestCSVExport` (3 tests) - CSV format validation for all result types
8. `TestIntegration` (2 tests) - End-to-end workflows (JSON and API paths)

### Test Fixtures

**File**: `fixtures/sample_traces_scrubbed.json`

**Creation Script**:
```python
import json

def scrub_trace(trace):
    """Remove sensitive data from trace."""
    scrubbed = trace.copy()
    scrubbed["inputs"] = {"__scrubbed__": True, "__note__": "Inputs removed for test fixture"}
    scrubbed["outputs"] = {"__scrubbed__": True, "__note__": "Outputs removed for test fixture"}
    return scrubbed

# Load original, scrub, save
with open("path/to/sample_traces_export.json") as f:
    data = json.load(f)

data["traces"] = [scrub_trace(t) for t in data["traces"]]

with open("fixtures/sample_traces_scrubbed.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Coverage Goals

- **Line coverage**: >90%
- **Branch coverage**: >85%
- **Critical paths**: 100% (data loading, percentile calculations, CSV export)

**Commands**:
```bash
# Run tests with coverage
pytest test_analyze_traces.py --cov=analyze_traces --cov-report=html

# Run specific test class
pytest test_analyze_traces.py::TestLatencyDistribution -v

# Run with verbose output
pytest test_analyze_traces.py -v -s
```

---

## Edge Cases & Error Handling

### 1. Data Quality Issues

**Empty/Incomplete Data**:
```python
if len(workflows) == 0:
    raise ValueError("No workflows found. Cannot perform analysis.")

if all(w.total_duration == 0 for w in workflows):
    raise ValueError("All workflows have zero duration. Cannot calculate latency.")
```

**Missing Timestamps**:
```python
if trace.start_time and not trace.end_time:
    logger.warning(f"Trace {trace.id} missing end_time. Excluding from analysis.")
    continue
```

**Malformed Hierarchies**:
```python
orphaned_traces = []
for trace in all_traces:
    if trace.parent_id and trace.parent_id not in trace_index:
        orphaned_traces.append(trace)
        logger.warning(f"Trace {trace.id} references missing parent {trace.parent_id}")
```

### 2. Statistical Edge Cases

**Small Sample Sizes**:
```python
MIN_SAMPLE_SIZE = 10

if len(workflows) < MIN_SAMPLE_SIZE:
    warnings.warn(
        f"Sample size ({len(workflows)}) below minimum ({MIN_SAMPLE_SIZE}). "
        "Results may not be statistically significant."
    )
```

**Outliers Dominating**:
```python
# Use both mean and median for robustness
# Use IQR for outlier detection
q1 = np.percentile(durations, 25)
q3 = np.percentile(durations, 75)
iqr = q3 - q1
outliers = [d for d in durations if d < q1 - 1.5*iqr or d > q3 + 1.5*iqr]
```

### 3. Parallel Execution Edge Cases

**Partial Validator Sets**:
```python
expected_validators = {"meta_evaluation_and_validation", "normative_validation", "simulated_testing"}
found = set(workflow.nodes.keys()) & expected_validators

if len(found) < 3:
    missing = expected_validators - found
    logger.warning(f"Workflow {workflow.root_trace.id} missing validators: {missing}")
    evidence_quality = "insufficient"
```

**Error-Driven Sequential**:
```python
def classify_execution_pattern(validators):
    """Distinguish intentional sequential from error-driven."""
    if any(v.error for v in validators):
        return "error_driven_sequential"

    start_deltas = [calculate_delta(validators)]
    return "parallel" if max(start_deltas) < 10 else "sequential"
```

### 4. File I/O Edge Cases

**Large Files**:
```python
import ijson

def load_from_json_streaming(filepath: str) -> TraceDataset:
    """Load large JSON files using streaming parser."""
    with open(filepath, 'rb') as f:
        parser = ijson.items(f, 'traces.item')
        for trace_dict in parser:
            # Process in batches
            ...
```

**Windows Path Issues**:
```python
from pathlib import Path

def load_from_json(filepath: str) -> TraceDataset:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
```

### 5. API Edge Cases

**Rate Limiting**:
```python
# Reuse exponential backoff from export_langsmith_traces.py
from export_langsmith_traces import LangSmithExporter

def load_from_langsmith_api(...):
    exporter = LangSmithExporter(api_key=api_key)
    # fetch_runs() already handles rate limiting
    runs = exporter.fetch_runs(project_name, limit)
```

**Incomplete API Response**:
```python
if not hasattr(run, 'child_runs') or run.child_runs is None:
    logger.warning(f"Run {run.id} missing child_runs. Using flat structure.")
    dataset.is_hierarchical = False
```

---

## Jupyter Notebook Structure

**File**: `notebooks/phase3a_performance_analysis.ipynb`

### Section 1: Setup & Data Loading

```python
# 1.1 Environment Setup
import sys
sys.path.append('..')  # Add parent directory to path

from analyze_traces import (
    load_from_json,
    load_from_langsmith_api,
    analyze_latency_distribution,
    identify_bottlenecks,
    verify_parallel_execution
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

# Configure display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_style("whitegrid")

# 1.2 Load Data (Choose approach)
# Option A: From JSON
dataset = load_from_json("path/to/sample_traces_export.json")

# Option B: From API (if hierarchical data needed)
# load_dotenv()
# dataset = load_from_langsmith_api(
#     api_key=os.getenv("LANGSMITH_API_KEY"),
#     project_name=os.getenv("LANGSMITH_PROJECT"),
#     limit=100
# )

print(f"Loaded {len(dataset.workflows)} workflows")
print(f"Hierarchical data: {dataset.is_hierarchical}")
```

### Section 2: Latency Distribution Analysis

```python
# 2.1 Calculate Percentiles
latency_results = analyze_latency_distribution(dataset.workflows)

# Display results table
df_latency = pd.DataFrame([
    {"Metric": "p50 (median)", "Value (min)": latency_results.p50_minutes,
     "Status": "✅" if 7 <= latency_results.p50_minutes <= 23 else "❌"},
    {"Metric": "p95", "Value (min)": latency_results.p95_minutes,
     "Status": "✅" if latency_results.p95_minutes <= 30 else "⚠️"},
    # ... more rows
])
display(df_latency)

# Export CSV
latency_results.to_csv("../output/latency_distribution.csv")

# 2.2 Visualization
durations = [w.total_duration / 60 for w in dataset.workflows]
plt.figure(figsize=(10, 6))
plt.hist(durations, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(7, color='green', linestyle='--', label='Claimed min')
plt.axvline(23, color='green', linestyle='--', label='Claimed max')
plt.axvline(latency_results.p50_minutes, color='blue', label=f'p50: {latency_results.p50_minutes:.1f}')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.title('Workflow Duration Distribution')
plt.legend()
plt.savefig('../output/latency_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Section 3: Bottleneck Identification

```python
# 3.1 Node Performance Analysis
bottleneck_results = identify_bottlenecks(dataset.workflows)

# Display table
df_bottlenecks = pd.DataFrame([
    {
        "Node": node.node_name,
        "Exec Count": node.execution_count,
        "Avg Duration (sec)": f"{node.avg_duration_seconds:.1f}",
        "% of Workflow": f"{node.avg_percent_of_workflow:.1f}%",
    }
    for node in sorted(bottleneck_results.node_performances,
                       key=lambda x: x.avg_duration_seconds,
                       reverse=True)
])
display(df_bottlenecks)

# Export CSV
bottleneck_results.to_csv("../output/bottleneck_analysis.csv")

# 3.2 Visualization
top_nodes = bottleneck_results.node_performances[:10]
node_names = [n.node_name for n in top_nodes]
durations = [n.avg_duration_seconds for n in top_nodes]

plt.figure(figsize=(12, 6))
plt.barh(node_names, durations, color='steelblue')
plt.xlabel('Average Duration (seconds)')
plt.title('Top 10 Slowest Nodes')
plt.tight_layout()
plt.savefig('../output/bottleneck_chart.png', dpi=300)
plt.show()
```

### Section 4: Parallel Execution Verification

```python
# 4.1 Validator Analysis
parallel_results = verify_parallel_execution(dataset.workflows)

print(f"Parallel Execution: {parallel_results.is_parallel} (Confidence: {parallel_results.confidence})")
print(f"Time savings: {parallel_results.avg_time_savings_seconds:.1f}s ({parallel_results.avg_time_savings_percent:.1f}%)")

# Export CSV
parallel_results.to_csv("../output/parallel_execution_analysis.csv")

# 4.2 Visualization
labels = ['Sequential\n(Hypothetical)', 'Parallel\n(Actual)']
times = [parallel_results.avg_sequential_time_seconds, parallel_results.avg_parallel_time_seconds]

plt.figure(figsize=(8, 6))
plt.bar(labels, times, color=['lightcoral', 'lightgreen'])
plt.ylabel('Time (seconds)')
plt.title('Validator Execution Time: Sequential vs Parallel')
savings = parallel_results.avg_time_savings_seconds
plt.annotate(f'Savings: {savings:.0f}s\n({parallel_results.avg_time_savings_percent:.1f}%)',
             xy=(0.5, max(times) * 0.5), ha='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='wheat'))
plt.savefig('../output/parallel_execution_savings.png', dpi=300)
plt.show()
```

### Section 5: Summary Report

```python
# Generate markdown summary
from analyze_traces import export_summary_markdown

export_summary_markdown(
    latency=latency_results,
    bottlenecks=bottleneck_results,
    parallel=parallel_results,
    filepath="../output/phase3a_performance_summary.md"
)

print("✅ Analysis complete!")
print("📁 Outputs saved to ../output/")
print("   - latency_distribution.csv")
print("   - bottleneck_analysis.csv")
print("   - parallel_execution_analysis.csv")
print("   - phase3a_performance_summary.md")
print("   - *.png (visualizations)")
```

---

## Validation Criteria (CHECK Phase)

### Functional Requirements
- ✅ Load both JSON export and API data successfully
- ✅ Calculate accurate p50/p95/p99 latency with known test data
- ✅ Identify primary bottleneck and rank nodes correctly
- ✅ Detect parallel vs sequential validator execution
- ✅ Export clean CSV files in documented format
- ✅ Jupyter notebook generates all visualizations

### Quality Requirements
- ✅ All 35+ tests pass with >90% line coverage, >85% branch coverage
- ✅ Code quality checks pass (black, ruff, mypy, bandit)
- ✅ Type-safe implementation with full type hints
- ✅ README documents analysis usage with examples

### Performance Requirements
- ✅ Analysis completes in <5 minutes for 100 workflows
- ✅ CSV files <10MB for reasonable datasets
- ✅ Memory usage reasonable for 1000 workflows

### Requirements Document Validation
- ✅ Latency Distribution: p50, p95, p99, outliers, claim validation
- ✅ Bottleneck Identification: node durations, % of workflow, variability
- ✅ Parallel Execution: time savings, confidence metrics

---

## Estimated Effort

**Total**: 12-15 hours over 4 days

**Breakdown**:
- Phase 1: Data loading (2-3 hours)
- Phase 2: Latency analysis (2-3 hours)
- Phase 3: Bottleneck analysis (3-4 hours)
- Phase 4: Parallel verification (2-3 hours)
- Phase 5: Output generation (2-3 hours)
- Phase 6: API loading - OPTIONAL (3-4 hours)
- Phase 7: Documentation & testing (2-3 hours)

---

## IMMEDIATE PREREQUISITE STEPS

**CRITICAL**: Before implementing analysis tool, validate child node retrieval:

### Step 1: Research LangSmith SDK (30-60 min)
- [ ] Review SDK documentation
- [ ] Test `client.read_run(run_id)` for single run
- [ ] Explore `get_run_tree()` or similar methods
- [ ] Check if `list_runs()` has parameters for children

### Step 2: Update Export Script (1-2 hours if needed)
- [ ] Modify `fetch_runs()` if SDK requires different approach
- [ ] Add `--include-children` flag (optional)
- [ ] Test with live API using `.env`

### Step 3: Test & Verify (30 min)
- [ ] Export 10-20 traces with hierarchy
- [ ] Verify `child_runs` populated
- [ ] Test pagination preserves children

### Step 4: Request Client Re-Export
- [ ] Provide updated script
- [ ] Request full dataset (100+ workflows)
- [ ] Verify new export has proper hierarchy

### Success Criteria for Prerequisite
- ✅ Export file contains populated `child_runs`
- ✅ Can see LangGraph → nodes → LLM hierarchy
- ✅ Pagination works for limits >100
- ✅ Sample size adequate (50+ workflows)

---

**End of Implementation Plan**

**Next Steps**: Complete prerequisite validation, then proceed with TDD implementation following this plan.
