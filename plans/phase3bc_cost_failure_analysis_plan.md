# Implementation Plan: Phase 3B & 3C Data Analysis

## Executive Summary

Implement test-driven Python analysis tools to calculate Phase 3B (Cost Analysis) and Phase 3C (Failure Pattern Analysis) metrics from LangSmith trace data. This extends the existing export-langsmith-data repository with two new analysis modules following established TDD patterns from Phase 3A.

**Key Architectural Decision**: Create separate modules (`analyze_cost.py` and `analyze_failures.py`) to maintain clean separation of concerns while reusing shared data structures from Phase 3A.

**Estimated Effort**: 18-26 hours (8-12 hours Phase 3B, 10-14 hours Phase 3C)

---

## Context

### Requirements from phase-3-data-analysis-outline.md

**Phase 3B: Cost Order-of-Magnitude Analysis**
- Calculate cost per workflow using token usage × Gemini 1.5 Pro pricing
- Breakdown costs by node type to identify expensive components
- Assess cache effectiveness (cost savings percentage)
- Project scaling costs at 10x, 100x, 1000x volume
- Determine economic viability at scale

**Phase 3C: Failure Pattern Analysis**
- Calculate overall success rate (no external wrapper data - infer from traces)
- Identify failure patterns by node and error type
- Detect and analyze retry sequences using heuristics
- Assess validator effectiveness and redundancy
- Quantify retry costs and success rates
- Identify quality risks at scale

### Data Availability Confirmed

✅ **Token Usage Data**: Available in traces as `usage_metadata` with fields:
- `input_tokens`, `output_tokens`, `total_tokens`
- `input_token_details.cache_read` for cached token counts

✅ **Trace Status**: Available for failure detection
- `status` field: "success", "error", etc.
- `error` field: Error messages for classification

❌ **No External Wrapper Data**: Must infer failures and retries from trace patterns

### Repository Architecture (Existing)

```
export-langsmith-data/
├── analyze_traces.py              # Phase 3A: 727 lines, performance analysis
├── test_analyze_traces.py         # 1,272 lines, 64 passing tests
├── export_langsmith_traces.py     # 697 lines, LangSmith API export
├── verify_analysis_report.py      # 366 lines, deterministic verification
└── (test files)                   # ~3,500 lines total test coverage
```

**Shared Data Structures** (from analyze_traces.py):
- `Trace`: Single run with id, name, status, duration, inputs, outputs, error
- `Workflow`: Complete execution with root_trace and hierarchical nodes
- `TraceDataset`: Container with workflows, metadata, hierarchical flag
- `load_from_json()`: Data loading function

---

## Implementation Strategy

### 1. Module Organization

**NEW Files to Create**:
```
export-langsmith-data/
├── analyze_cost.py                # NEW - Phase 3B Cost Analysis (~700 lines)
├── test_analyze_cost.py           # NEW - Phase 3B tests (~900 lines, ~45 tests)
├── analyze_failures.py            # NEW - Phase 3C Failure Analysis (~800 lines)
└── test_analyze_failures.py       # NEW - Phase 3C tests (~1000 lines, ~56 tests)
```

**MODIFIED Files**:
```
└── verify_analysis_report.py      # Extend with 3B/3C verification functions
```

**Rationale**: Separate modules maintain clean boundaries, enable independent evolution, and keep test suites focused. Each phase has distinct concerns (economics vs quality) that warrant separate files.

---

## Phase 3B: Cost Analysis Implementation

### Configuration Constants

```python
# analyze_cost.py - Top of file

from dataclasses import dataclass

@dataclass
class PricingConfig:
    """Configurable pricing model for any LLM provider."""
    model_name: str
    input_tokens_per_1k: float           # Cost per 1K input tokens
    output_tokens_per_1k: float          # Cost per 1K output tokens
    cache_read_per_1k: Optional[float] = None   # Cost per 1K cache read tokens (if applicable)

    def __post_init__(self):
        """Validate pricing configuration."""
        if self.input_tokens_per_1k < 0 or self.output_tokens_per_1k < 0:
            raise ValueError("Token prices must be non-negative")
        if self.cache_read_per_1k is not None and self.cache_read_per_1k < 0:
            raise ValueError("Cache read price must be non-negative")


# Example pricing configs for reference (NOT hard-coded defaults)
# Users should create their own PricingConfig instances
EXAMPLE_PRICING_CONFIGS = {
    "gemini_1.5_pro": PricingConfig(
        model_name="Gemini 1.5 Pro",
        input_tokens_per_1k=0.00125,      # $1.25 per 1M input tokens
        output_tokens_per_1k=0.005,       # $5.00 per 1M output tokens
        cache_read_per_1k=0.0003125,      # $0.3125 per 1M cache read tokens
    ),
    # Add other providers as examples
}

SCALING_FACTORS = [1, 10, 100, 1000]  # Current, 10x, 100x, 1000x
```

### Core Data Structures

```python
@dataclass
class TokenUsage:
    """Token usage for a single trace."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: Optional[int] = None      # From input_token_details.cache_read

@dataclass
class CostBreakdown:
    """Cost breakdown for a single trace."""
    trace_id: str
    trace_name: str
    input_cost: float
    output_cost: float
    cache_cost: float
    total_cost: float
    token_usage: TokenUsage

@dataclass
class NodeCostSummary:
    """Cost summary for a node type across workflows."""
    node_name: str
    execution_count: int
    total_cost: float
    avg_cost_per_execution: float
    percent_of_total_cost: float

@dataclass
class CostAnalysisResults:
    """Complete cost analysis results."""
    # Per-workflow metrics
    avg_cost_per_workflow: float
    median_cost_per_workflow: float
    min_cost: float
    max_cost: float

    # Node-level breakdown
    node_summaries: List[NodeCostSummary]    # Sorted by cost descending
    top_cost_driver: Optional[str]

    # Cache effectiveness
    cache_effectiveness_percent: Optional[float]
    cache_savings_dollars: Optional[float]

    # Scaling projections
    scaling_projections: Dict[str, ScalingProjection]  # "10x", "100x", etc.

    # Metadata
    total_workflows_analyzed: int
    data_quality_notes: List[str]

    def to_csv(self) -> str:
        """Export to CSV format for reporting."""
```

### Key Functions

**Token Extraction**:
```python
def extract_token_usage(trace: Trace) -> Optional[TokenUsage]:
    """
    Extract token usage from trace outputs/inputs.

    Checks:
    1. trace.outputs["usage_metadata"]
    2. trace.inputs["usage_metadata"] (fallback)
    3. Extracts cache_read from input_token_details if available
    """

def extract_workflow_tokens(workflow: Workflow) -> Dict[str, TokenUsage]:
    """Extract token usage for all traces in workflow."""
```

**Cost Calculation**:
```python
def calculate_trace_cost(
    token_usage: TokenUsage,
    pricing_config: PricingConfig
) -> CostBreakdown:
    """Calculate cost for single trace using pricing model."""

def calculate_workflow_cost(
    workflow: Workflow,
    pricing_config: PricingConfig
) -> Optional[WorkflowCostAnalysis]:
    """Calculate total cost and breakdown by node."""
```

**Aggregation & Analysis**:
```python
def aggregate_node_costs(
    workflow_analyses: List[WorkflowCostAnalysis]
) -> List[NodeCostSummary]:
    """Aggregate costs by node type, sorted by total_cost descending."""

def calculate_cache_effectiveness(
    workflow_analyses: List[WorkflowCostAnalysis]
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate cache effectiveness if cache data available.
    Returns (effectiveness_percent, cost_without_cache, savings_dollars).
    """

def project_scaling_costs(
    avg_cost_per_workflow: float,
    current_workflow_count: int,
    scaling_factors: List[int]
) -> Dict[str, ScalingProjection]:
    """Project costs at 10x, 100x, 1000x scale."""
```

**Main Entry Point**:
```python
def analyze_costs(
    workflows: List[Workflow],
    pricing_config: PricingConfig
) -> CostAnalysisResults:
    """
    Perform complete cost analysis on workflows.

    Args:
        workflows: List of Workflow objects to analyze
        pricing_config: PricingConfig with provider-specific rates

    Main entry point for Phase 3B.

    Example:
        # Create custom pricing config
        pricing = PricingConfig(
            model_name="Gemini 1.5 Pro",
            input_tokens_per_1k=0.00125,
            output_tokens_per_1k=0.005,
            cache_read_per_1k=0.0003125
        )

        results = analyze_costs(workflows, pricing)
    """
```

### Testing Strategy (Phase 3B)

**Test File**: `test_analyze_cost.py` (~45 tests)

**Test Classes**:
1. `TestTokenExtraction` - Extract tokens from various trace structures
2. `TestCostCalculation` - Basic and edge case cost calculations
3. `TestNodeCostAggregation` - Multi-workflow aggregation
4. `TestCacheEffectiveness` - Cache savings calculations
5. `TestScalingProjections` - Scaling math and viability thresholds
6. `TestCostAnalysisIntegration` - End-to-end with mock workflows
7. `TestCSVExport` - CSV formatting

**Key Test Cases**:
- Token extraction from `outputs["usage_metadata"]`
- Token extraction with cache_read tokens
- Missing token data returns None gracefully
- Zero-cost traces handled correctly
- Cost calculation accuracy (verify pricing math)
- Node aggregation sorted correctly
- Scaling projections calculate monthly costs
- CSV format matches expected structure

**TDD Workflow**: Write test → See it fail → Implement minimal code → Pass → Refactor

### CSV Exports

```python
def export_node_costs_csv(node_summaries: List[NodeCostSummary]) -> str:
    """Export node cost breakdown to CSV."""

def export_scaling_projections_csv(projections: Dict[str, ScalingProjection]) -> str:
    """Export scaling projections to CSV."""
```

---

## Phase 3C: Failure Pattern Analysis Implementation

### Configuration Constants

```python
# analyze_failures.py - Top of file

# Status values indicating failure
FAILURE_STATUSES = {"error", "failed", "cancelled"}
SUCCESS_STATUSES = {"success"}

# Retry detection heuristics
RETRY_DETECTION_CONFIG = {
    "max_time_window_seconds": 300,  # 5 min window for retry detection
    "same_node_threshold": 2,        # 2+ executions = potential retry
}

# Validator node names (from Phase 3A)
VALIDATOR_NODES = {
    "meta_evaluation_and_validation",
    "normative_validation",
    "simulated_testing",
}

# Error classification patterns (regex)
ERROR_PATTERNS = {
    "validation_failure": r"validation.*fail|invalid.*spec",
    "api_timeout": r"timeout|timed out",
    "import_error": r"import.*fail|upload.*fail",
    "llm_error": r"model.*error|generation.*fail|token.*limit",
    "unknown": r".*",  # Catch-all
}
```

### Core Data Structures

```python
@dataclass
class FailureInstance:
    """Single failure occurrence."""
    trace_id: str
    trace_name: str
    workflow_id: str
    error_message: Optional[str]
    error_type: str                  # Classified from ERROR_PATTERNS
    timestamp: Optional[datetime]

@dataclass
class RetrySequence:
    """Detected retry sequence."""
    node_name: str
    workflow_id: str
    attempt_count: int
    attempts: List[Trace]            # Ordered by start_time
    final_status: str                # 'success' or 'failed'
    total_duration_seconds: float
    total_cost_estimate: Optional[float] = None

@dataclass
class NodeFailureStats:
    """Failure statistics for a node type."""
    node_name: str
    total_executions: int
    failure_count: int
    success_count: int
    failure_rate_percent: float
    retry_sequences_detected: int
    avg_retries_when_failing: float
    common_error_types: Dict[str, int]  # error_type -> count

@dataclass
class ValidatorEffectivenessAnalysis:
    """Validator effectiveness assessment."""
    validator_name: str
    total_executions: int
    caught_issues_count: int         # Failures detected
    pass_rate_percent: float
    is_necessary: bool               # Based on redundancy analysis

@dataclass
class FailureAnalysisResults:
    """Complete failure pattern analysis results."""
    # Overall metrics
    total_workflows: int
    successful_workflows: int
    failed_workflows: int
    overall_success_rate_percent: float

    # Node-level breakdown
    node_failure_stats: List[NodeFailureStats]  # Sorted by failure_rate
    highest_failure_node: Optional[str]

    # Error distribution
    error_type_distribution: Dict[str, int]
    most_common_error_type: Optional[str]

    # Retry analysis
    total_retry_sequences: int
    retry_sequences: List[RetrySequence]
    retry_success_rate_percent: Optional[float]
    avg_cost_of_retries: Optional[float]

    # Validator analysis
    validator_analyses: List[ValidatorEffectivenessAnalysis]
    redundant_validators: List[str]

    # Quality risks
    quality_risks_at_scale: List[str]

    def to_csv(self) -> str:
        """Export to CSV format for reporting."""
```

### Key Functions

**Failure Detection**:
```python
def detect_failures(workflow: Workflow) -> List[FailureInstance]:
    """Detect all failures in workflow using trace.status and trace.error."""

def classify_error(error_message: Optional[str]) -> str:
    """Classify error into type using regex patterns."""
```

**Retry Detection**:
```python
def detect_retry_sequences(workflow: Workflow) -> List[RetrySequence]:
    """
    Detect retry sequences using heuristics:
    - Multiple executions of same node within time window
    - Ordered by start_time
    """

def calculate_retry_success_rate(
    retry_sequences: List[RetrySequence]
) -> Optional[float]:
    """Calculate % of retries that eventually succeed."""
```

**Node & Validator Analysis**:
```python
def analyze_node_failures(
    workflows: List[Workflow]
) -> List[NodeFailureStats]:
    """Analyze failure patterns by node type."""

def analyze_validator_effectiveness(
    workflows: List[Workflow]
) -> List[ValidatorEffectivenessAnalysis]:
    """Assess whether all 3 validators are necessary."""

def detect_validator_redundancy(
    validator_analyses: List[ValidatorEffectivenessAnalysis]
) -> List[str]:
    """Identify validators with >90% overlap in caught issues."""
```

**Root Cause & Risk**:
```python
def identify_root_causes(
    failure_instances: List[FailureInstance],
    node_stats: List[NodeFailureStats]
) -> Dict[str, List[str]]:
    """
    Categorize root causes:
    - Prompt issues (validation failures)
    - Logic bugs (repeated errors)
    - External failures (API timeouts, import errors)
    """

def assess_quality_risks_at_scale(
    results: FailureAnalysisResults,
    current_volume: int,
    projected_volume: int
) -> List[str]:
    """Generate risk warnings for scaling."""
```

**Main Entry Point**:
```python
def analyze_failures(workflows: List[Workflow]) -> FailureAnalysisResults:
    """
    Perform complete failure pattern analysis.
    Main entry point for Phase 3C.
    """
```

### Testing Strategy (Phase 3C)

**Test File**: `test_analyze_failures.py` (~56 tests)

**Test Classes**:
1. `TestFailureDetection` - Detect failures from status/error fields
2. `TestRetryDetection` - Heuristic-based retry sequence detection
3. `TestNodeFailureAnalysis` - Node-level statistics
4. `TestValidatorEffectiveness` - Validator redundancy analysis
5. `TestRootCauseAnalysis` - Error categorization
6. `TestQualityRiskAssessment` - Risk identification
7. `TestFailureAnalysisIntegration` - End-to-end tests
8. `TestCSVExport` - CSV formatting

**Key Test Cases**:
- Detect single and multiple failures in workflow
- Classify errors using regex patterns
- Detect retry sequences with 2, 3, 5+ attempts
- Respect time window for retry detection
- Calculate retry success rates correctly
- Identify redundant validators (overlap detection)
- Generate appropriate risk warnings
- Handle workflows with 0% and 100% success rates

### CSV Exports

```python
def export_node_failures_csv(node_stats: List[NodeFailureStats]) -> str:
    """Export node failure statistics to CSV."""

def export_validator_effectiveness_csv(
    analyses: List[ValidatorEffectivenessAnalysis]
) -> str:
    """Export validator effectiveness to CSV."""

def export_retry_analysis_csv(retry_sequences: List[RetrySequence]) -> str:
    """Export retry sequence analysis to CSV."""
```

---

## Integration with Verification Tool

### Extend verify_analysis_report.py

Add two new verification functions:

```python
def verify_cost_analysis(
    dataset: TraceDataset,
    expected: Optional[Dict[str, Any]] = None
) -> CostAnalysisResults:
    """
    Verify Phase 3B cost calculations.

    Displays:
    - Cost per workflow (avg, median, range)
    - Top 3 cost drivers
    - Scaling projections (10x, 100x, 1000x)
    - Cache effectiveness if available
    """

def verify_failure_analysis(
    dataset: TraceDataset,
    expected: Optional[Dict[str, Any]] = None
) -> FailureAnalysisResults:
    """
    Verify Phase 3C failure calculations.

    Displays:
    - Overall success rate
    - Top 5 nodes by failure rate
    - Retry analysis (sequences detected, success rate)
    - Validator effectiveness
    """
```

Update `main()` to accept `--phases` argument:
```python
parser.add_argument("--phases", type=str, default="all",
                   help="Phases to verify: 3a, 3b, 3c, or all")
```

---

## Implementation Sequence (TDD Workflow)

### Phase 3B: Cost Analysis (8-12 hours)

**Step 1: Setup** (30 min)
- Create `analyze_cost.py` with pricing constants and imports
- Create `test_analyze_cost.py` with test structure
- Add initial docstrings

**Step 2: Token Extraction** (2-3 hours)
1. Write test: `test_extract_token_usage_from_outputs()`
2. Implement `extract_token_usage()` to pass
3. Write test: `test_extract_token_usage_with_cache_data()`
4. Extend implementation for cache tokens
5. Write test: `test_extract_token_usage_missing_data()`
6. Handle None return gracefully
7. **CRITICAL**: Test with REAL trace data first to confirm field locations

**Step 3: Cost Calculation** (2-3 hours)
1. Write test: `test_calculate_trace_cost_basic()`
2. Implement basic calculation (input + output tokens)
3. Write test: `test_calculate_trace_cost_with_cache()`
4. Add cache cost logic
5. Write test: `test_calculate_workflow_cost()`
6. Implement workflow-level aggregation

**Step 4: Aggregation & Analysis** (2-3 hours)
1. Write tests for `aggregate_node_costs()`
2. Implement node-level aggregation with sorting
3. Write tests for `calculate_cache_effectiveness()`
4. Implement cache savings calculation
5. Write tests for `project_scaling_costs()`
6. Implement scaling projections

**Step 5: Main Analysis & Export** (1-2 hours)
1. Write integration test for `analyze_costs()`
2. Implement main orchestration function
3. Add data quality checks and warnings
4. Implement CSV export functions
5. Test CSV format

**Step 6: Verification Integration** (1 hour)
1. Add `verify_cost_analysis()` to verify_analysis_report.py
2. Test verification with sample data
3. Update main() with --phases argument

### Phase 3C: Failure Analysis (10-14 hours)

**Step 1: Setup** (30 min)
- Create `analyze_failures.py` with error patterns and constants
- Create `test_analyze_failures.py` with test structure

**Step 2: Failure Detection** (2-3 hours)
1. Write tests for `detect_failures()`
2. Implement status-based detection
3. Write tests for `classify_error()`
4. Implement regex-based classification

**Step 3: Retry Detection** (3-4 hours)
1. Write tests for simple retry detection (2 attempts)
2. Implement basic retry detection
3. Write tests for complex scenarios (5+ attempts, time windows)
4. Refine detection heuristics
5. Write tests for `calculate_retry_success_rate()`
6. Implement success rate calculation

**Step 4: Node Failure Analysis** (2-3 hours)
1. Write tests for `analyze_node_failures()`
2. Implement aggregation and statistics
3. Test sorting by failure_rate
4. Test error type aggregation

**Step 5: Validator Analysis** (2-3 hours)
1. Write tests for `analyze_validator_effectiveness()`
2. Implement effectiveness metrics
3. Write tests for `detect_validator_redundancy()`
4. Implement overlap detection logic

**Step 6: Root Cause & Integration** (2-3 hours)
1. Write tests for `identify_root_causes()`
2. Implement categorization logic
3. Write tests for `assess_quality_risks_at_scale()`
4. Implement risk warning generation
5. Write integration test for `analyze_failures()`
6. Implement main function and CSV exports

**Step 7: Verification Integration** (1 hour)
1. Add `verify_failure_analysis()` to verify_analysis_report.py
2. Test verification with sample data

---

## Data Quality Handling

### Edge Cases to Handle

**Phase 3B**:
- Missing token data → Skip trace, add to data_quality_notes
- Zero-cost traces → Include in analysis (valid for non-LLM nodes)
- Missing cache data → Skip cache analysis, report "N/A"
- Incomplete workflows → Partial cost calculated, note in quality report

**Phase 3C**:
- No failures → Report 100% success rate (valid result)
- Missing error messages → Classify as "unknown"
- Ambiguous retries → Use time window + node name heuristic
- Missing validators → Note in analysis, skip redundancy check

### Data Quality Reporting

Both modules include `data_quality_notes` field in results:

```python
data_quality_notes: List[str] = [
    "Token data available for 95/100 workflows",
    "Cache data not available - cache analysis skipped",
    "Retry detection based on heuristics (no definitive markers)",
]
```

---

## Critical Pre-Implementation Steps

### BEFORE starting implementation, MUST complete:

1. **Verify Token Data Structure** (30 min)
   ```python
   # Inspect actual trace structure
   import json
   with open("trace_export_1000.json") as f:
       data = json.load(f)

   # Find LLM trace and print structure
   trace = [t for t in data["traces"] if t["name"] == "ChatGoogleGenerativeAI"][0]
   print(json.dumps(trace["outputs"], indent=2))
   ```

2. **Create Client-Specific Pricing Script** (15 min)
   - Research current Gemini 1.5 Pro pricing (Dec 2025)
   - Create `scripts/analyze_with_gemini_pricing.py` in client repo
   - Script instantiates PricingConfig with Gemini rates
   - Calls analyze_costs() with custom pricing

3. **Document Field Locations** (15 min)
   - Record exact path to usage_metadata
   - Record exact path to cache_read tokens
   - Create example trace structure in docstrings

---

## Success Criteria

### Phase 3B Complete When:
- ✅ All 45+ tests passing
- ✅ Can answer: "What does a workflow cost?" → $__
- ✅ Can answer: "What's the biggest cost driver?" → [node name]
- ✅ Can answer: "Can it scale to 100x volume?" → YES/NO with projection
- ✅ Cache effectiveness calculated (or "N/A" if unavailable)
- ✅ CSV exports generated for all metrics
- ✅ Verification tool confirms calculations

### Phase 3C Complete When:
- ✅ All 56+ tests passing
- ✅ Can answer: "What's the success rate?" → ___%
- ✅ Can answer: "Where do failures happen most?" → [node name + rate]
- ✅ Can answer: "Are retries effective?" → ___% success rate
- ✅ Can answer: "Are all 3 validators necessary?" → YES/NO with data
- ✅ CSV exports generated for failures, retries, validators
- ✅ Quality risks identified for scaling
- ✅ Verification tool confirms calculations

---

## Deliverables

### Code Deliverables
1. `analyze_cost.py` (~700 lines) with full documentation
2. `test_analyze_cost.py` (~900 lines, 45+ tests passing)
3. `analyze_failures.py` (~800 lines) with full documentation
4. `test_analyze_failures.py` (~1000 lines, 56+ tests passing)
5. Extended `verify_analysis_report.py` with 3B/3C verification
6. Updated README.md with Phase 3B/3C usage examples

### Analysis Outputs (for client Assessment)
1. Cost analysis CSV exports:
   - Node cost breakdown
   - Scaling projections
   - Cache effectiveness (if available)
2. Failure analysis CSV exports:
   - Node failure statistics
   - Retry analysis
   - Validator effectiveness
3. Verification reports confirming all calculations

---

## Repository Separation Strategy

**Generalized Tools** (export-langsmith-data repo):
- `analyze_cost.py` - Generic cost analysis for any LangSmith traces
- `analyze_failures.py` - Generic failure analysis for any traces
- All test files - Reusable test patterns
- `verify_analysis_report.py` - Verification framework

**Client-Specific Analysis** (client-project/Assessment/data):
- Actual trace data files (.json)
- Generated CSV reports
- Client-specific interpretation and findings
- Phase 3B/3C markdown reports referencing data
- **Custom pricing script** (`scripts/analyze_with_gemini_pricing.py`):
  - Imports `PricingConfig` from generalized tool
  - Creates config with Gemini-specific rates
  - Runs analysis with client's trace data

This separation allows:
- Reuse of analysis tools across projects
- Client confidentiality (data stays in Assessment repo)
- Test-first development in open repo
- Easy updates to pricing models and error patterns

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Token data not in expected location | Inspect real data FIRST, implement flexible extraction |
| Cache data unavailable | Make cache analysis optional, graceful degradation |
| Retry detection false positives | Tune time window (5 min), manual spot-checking |
| Pricing model outdated | Externalize constants, document update procedure |
| Large datasets cause memory issues | Process in batches if needed (unlikely with n=10-1000) |
| Test coverage gaps | Strict TDD, aim for >90% coverage |

---

## Critical Files

**To Read** (for reference patterns):
1. `analyze_traces.py`
2. `test_analyze_traces.py`
3. `verify_analysis_report.py`

**To Create**:
1. `analyze_cost.py`
2. `test_analyze_cost.py`
3. `analyze_failures.py`
4. `test_analyze_failures.py`

**To Modify**:
1. `verify_analysis_report.py`

**Data Files** (for inspection):
1. `trace_export_1000.json` (newly exported)
2. `../client-project/Assessment/data/trace_export_1000.json`
3. `../client-project/Assessment/data/trace_export_complete_workflows.json`

---

## Next Steps After Approval

1. **Data Investigation** (30 min) - Inspect token field locations in real traces
2. **Pricing Research** (15 min) - Confirm Gemini 1.5 Pro current pricing
3. **Setup** (30 min) - Create files, initial structure, constants
4. **Phase 3B Implementation** (8-12 hours) - TDD workflow
5. **Phase 3C Implementation** (10-14 hours) - TDD workflow
6. **Verification & Documentation** (2-3 hours) - Final testing, README updates
7. **Generate Reports** (1-2 hours) - Run on actual data, export CSVs for Assessment

**Total Estimated Time**: 18-26 hours for complete implementation
