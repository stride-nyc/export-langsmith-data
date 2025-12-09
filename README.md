# LangSmith Data Export & Analysis Tools

Export and analyze workflow trace data from LangSmith projects for performance insights and optimization.

## Overview

This toolkit provides comprehensive capabilities for LangSmith trace analysis:
1. **Data Export** (`export_langsmith_traces.py`) - Export trace data from LangSmith using the SDK API
2. **Performance Analysis** (`analyze_traces.py`) - Analyze exported traces for latency, bottlenecks, and parallel execution (Phase 3A)
3. **Cost Analysis** (`analyze_cost.py`) - Calculate workflow costs with configurable pricing models (Phase 3B)
4. **Failure Pattern Analysis** (`analyze_failures.py`) - Detect failures, retry sequences, and error patterns (Phase 3C)

Designed for users on Individual Developer plans without bulk export features, with robust error handling, rate limiting, and comprehensive analysis capabilities. All modules follow strict TDD methodology with 99+ tests and full type safety.

## Features

### Data Export (`export_langsmith_traces.py`)
- Export N most recent traces from any LangSmith project
- **Automatic pagination** - Handles large exports (> 100 records) seamlessly with progress indication
- **Environment variable support** - Configure once via `.env` file for simplified usage
- **Hierarchical data export** - Includes child_runs with `--include-children` flag
- Automatic rate limiting with exponential backoff
- Progress indication for long-running exports
- Comprehensive error handling (auth, network, rate limits)
- Structured JSON output with metadata
- Type-safe implementation with full type hints
- Test-driven development with pytest suite (33 tests, high coverage)

### Performance Analysis (`analyze_traces.py`)
- **Latency Distribution Analysis** - Calculate p50/p95/p99 percentiles, identify outliers
- **Bottleneck Identification** - Rank nodes by execution time, identify primary bottlenecks
- **Parallel Execution Verification** - Detect parallel vs sequential execution, calculate time savings
- **CSV Export** - Export analysis results to CSV files for reporting
- **Interactive Jupyter Notebook** - Visual analysis workflow with automated reporting
- 31 comprehensive tests, 100% type-safe with mypy strict mode

## Requirements

- Python 3.8 or higher
- LangSmith API key (Individual Developer plan or higher)
- Virtual environment manager (uv or venv)
- Additional packages: numpy (for analysis), jupyter (for notebook)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd export-langsmith-data
```

### 2. Set up virtual environment

**Option A: Using uv (recommended)**
```bash
uv venv
.venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # Linux/Mac
```

**Option B: Using venv**
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # Linux/Mac
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
# or
pip install -r requirements.txt
```

### 4. Configure API key

Create a `.env` file from the template:
```bash
cp .env.example .env
```

Edit `.env` and add your LangSmith API key:
```env
LANGSMITH_API_KEY=lsv2_pt_your_api_key_here
```

Get your API key from: https://smith.langchain.com/settings

## Usage

## Part 1: Data Export

### Option 1: Using Environment Variables (Recommended)

Set up your `.env` file once:

```bash
cp .env.example .env
# Edit .env with your values:
# LANGSMITH_API_KEY=lsv2_pt_your_key_here
# LANGSMITH_PROJECT=your-project-name
# LANGSMITH_LIMIT=150
```

Then use simple commands:

```bash
python export_langsmith_traces.py --output "traces_export.json"
```

### Option 2: Using Command Line Arguments

```bash
python export_langsmith_traces.py \
  --api-key "lsv2_pt_..." \
  --project "your-project-name" \
  --limit 150 \
  --output "traces_export.json"
```

### Option 3: Mixed Usage (Override Environment Variables)

```bash
# Override just the project while using env vars for api-key and limit
python export_langsmith_traces.py \
  --project "different-project" \
  --output "traces_export.json"
```

### Parameters

- `--api-key` (optional): LangSmith API key for authentication (default: `LANGSMITH_API_KEY` env var)
- `--project` (optional): LangSmith project name or ID (default: `LANGSMITH_PROJECT` env var)
- `--limit` (optional): Number of most recent traces to export (default: `LANGSMITH_LIMIT` env var)
  - For limits > 100, the tool automatically handles pagination across multiple API calls
  - If fewer records exist in the project, you'll receive a warning and all available records
- `--output` (required): Output JSON file path

**Note**: While the CLI arguments are now optional, the values must be provided either via command line or environment variables.

### Examples

**Using environment variables:**
```bash
# Set up .env file once
echo "LANGSMITH_API_KEY=lsv2_pt_abc123..." >> .env
echo "LANGSMITH_PROJECT=your-project-name" >> .env
echo "LANGSMITH_LIMIT=200" >> .env

# Simple usage
python export_langsmith_traces.py --output "traces_2025-11-28.json"
```

**Using CLI arguments:**
```bash
python export_langsmith_traces.py \
  --api-key "lsv2_pt_abc123..." \
  --project "your-project-name" \
  --limit 200 \
  --output "traces_2025-11-28.json"
```

**Mixed usage:**
```bash
# Use env vars for api-key and project, override limit
python export_langsmith_traces.py \
  --limit 500 \
  --output "large_export.json"
```

## Pagination for Large Exports

The LangSmith API limits results to 100 records per call. This tool automatically handles pagination for larger exports with progress indication:

**Example: Exporting 500 records**
```bash
python export_langsmith_traces.py --limit 500 --output large_export.json
```

**Output:**
```
🚀 Exporting 500 traces from project 'my-project'...
✓ Connected to LangSmith API
📥 Fetching traces...
  📄 Fetching 500 runs across 5 pages...
    ✓ Page 1/5: 100 runs (Total: 100)
    ✓ Page 2/5: 100 runs (Total: 200)
    ✓ Page 3/5: 100 runs (Total: 300)
    ✓ Page 4/5: 100 runs (Total: 400)
    ✓ Page 5/5: 100 runs (Total: 500)
✓ Fetched 500 traces
🔄 Formatting trace data...
✓ Data formatted
💾 Exporting to large_export.json...
✅ Export complete! Saved to large_export.json
```

**If Project Has Fewer Records:**
```
⚠️  Warning: Fetched 250 runs (requested 500)
```

**Pagination Features:**
- Automatic chunking into 100-record pages
- Progress indication for multi-page exports
- Rate limiting between pages (500ms delay)
- Retry logic per page for reliability
- Warning when fewer records available than requested

## Part 2: Performance Analysis

Once you have exported trace data, use the analysis tools to gain performance insights.

### Quick Start with Jupyter Notebook (Recommended)

1. **Export trace data with hierarchical information:**
   ```bash
   python export_langsmith_traces.py \
     --limit 100 \
     --output traces_export.json \
     --include-children
   ```

2. **Launch Jupyter notebook:**
   ```bash
   jupyter notebook notebooks/langsmith_trace_performance_analysis.ipynb
   ```

3. **Update the file path** in cell 2 to point to your export file

4. **Run all cells** to generate:
   - Latency distribution metrics (p50/p95/p99)
   - Bottleneck analysis with node rankings
   - Parallel execution verification
   - CSV exports in `output/` directory

### Verifying Analysis Results

After generating analysis results, use the verification tool to ensure accuracy:

```bash
# Basic verification - Phase 3A only (default)
python verify_analysis_report.py traces_export.json

# Verify all phases (3A + 3B + 3C)
python verify_analysis_report.py traces_export.json --phases all

# Verify specific phases
python verify_analysis_report.py traces_export.json --phases 3b
python verify_analysis_report.py traces_export.json --phases 3c
python verify_analysis_report.py traces_export.json --phases "3a,3b"

# Verify against expected values
python verify_analysis_report.py traces_export.json --expected-values expected.json

# Use custom pricing model for cost analysis
python verify_analysis_report.py traces_export.json --phases 3b --pricing-model gemini_1.5_pro
```

The verification tool:
- Regenerates all calculations from raw data
- Provides deterministic verification of findings
- Optionally compares against expected values (PASS/FAIL indicators)
- Supports selective phase verification (3a, 3b, 3c, or all)
- Useful for auditing and validating reports

**Example expected values JSON:**
```json
{
  "sample_size": 10,
  "latency": {
    "p50": 25.25,
    "p95": 46.03,
    "mean": 26.23
  },
  "parallel": {
    "parallel_pct": 30.0,
    "savings_s": 201.5
  }
}
```

### Cost Analysis (Phase 3B)

Analyze workflow costs based on token usage with configurable pricing models:

```python
from analyze_cost import (
    analyze_costs,
    PricingConfig,
    EXAMPLE_PRICING_CONFIGS,
)
from analyze_traces import load_from_json

# Load exported trace data
dataset = load_from_json("traces_export.json")

# Option 1: Use example pricing config
pricing = EXAMPLE_PRICING_CONFIGS["gemini_1.5_pro"]

# Option 2: Create custom pricing config
pricing = PricingConfig(
    model_name="Custom Model",
    input_tokens_per_1k=0.001,      # $1.00 per 1M input tokens
    output_tokens_per_1k=0.003,     # $3.00 per 1M output tokens
    cache_read_per_1k=0.0001,       # $0.10 per 1M cache read tokens (optional)
)

# Run cost analysis
results = analyze_costs(
    workflows=dataset.workflows,
    pricing_config=pricing,
    scaling_factors=[1, 10, 100, 1000],  # Optional, defaults to [1, 10, 100, 1000]
    monthly_workflow_estimate=10000,     # Optional, for monthly cost projections
)

# Access results
print(f"Average cost per workflow: ${results.avg_cost_per_workflow:.4f}")
print(f"Median cost: ${results.median_cost_per_workflow:.4f}")
print(f"Top cost driver: {results.top_cost_driver}")

# View node-level breakdown
for node in results.node_summaries[:3]:  # Top 3 nodes
    print(f"  {node.node_name}:")
    print(f"    Total cost: ${node.total_cost:.4f}")
    print(f"    Executions: {node.execution_count}")
    print(f"    Avg per execution: ${node.avg_cost_per_execution:.6f}")
    print(f"    % of total: {node.percent_of_total_cost:.1f}%")

# View scaling projections
for scale_label, projection in results.scaling_projections.items():
    print(f"{scale_label}: ${projection.total_cost:.2f} for {projection.workflow_count} workflows")
    if projection.cost_per_month_30days:
        print(f"  Monthly estimate: ${projection.cost_per_month_30days:.2f}/month")
```

**Cost Analysis Features:**
- Configurable pricing for any LLM provider (not hard-coded)
- Token usage extraction (input/output/cache tokens)
- Workflow-level cost aggregation
- Node-level cost breakdown with percentages
- Scaling projections at 1x, 10x, 100x, 1000x volume
- Optional monthly cost estimates
- Data quality reporting for missing token data

### Failure Pattern Analysis (Phase 3C)

Detect and analyze failure patterns, retry sequences, and error distributions:

```python
from analyze_failures import (
    analyze_failures,
    FAILURE_STATUSES,
    ERROR_PATTERNS,
)
from analyze_traces import load_from_json

# Load exported trace data
dataset = load_from_json("traces_export.json")

# Run failure analysis
results = analyze_failures(workflows=dataset.workflows)

# Overall metrics
print(f"Total workflows: {results.total_workflows}")
print(f"Success rate: {results.overall_success_rate_percent:.1f}%")
print(f"Failed workflows: {results.failed_workflows}")

# Node failure breakdown
print("\nTop 5 nodes by failure rate:")
for node in results.node_failure_stats[:5]:
    print(f"  {node.node_name}:")
    print(f"    Failure rate: {node.failure_rate_percent:.1f}%")
    print(f"    Failures: {node.failure_count}/{node.total_executions}")
    print(f"    Retry sequences: {node.retry_sequences_detected}")
    print(f"    Common errors: {node.common_error_types}")

# Error distribution
print("\nError type distribution:")
for error_type, count in results.error_type_distribution.items():
    print(f"  {error_type}: {count}")

# Retry analysis
print(f"\nTotal retry sequences detected: {results.total_retry_sequences}")
if results.retry_success_rate_percent:
    print(f"Retry success rate: {results.retry_success_rate_percent:.1f}%")

# Example retry sequence details
for retry_seq in results.retry_sequences[:3]:  # First 3 retry sequences
    print(f"\nRetry sequence in {retry_seq.node_name}:")
    print(f"  Attempts: {retry_seq.attempt_count}")
    print(f"  Final status: {retry_seq.final_status}")
    print(f"  Total duration: {retry_seq.total_duration_seconds:.1f}s")
```

**Failure Analysis Features:**
- Status-based failure detection (error, failed, cancelled)
- Regex-based error classification (validation, timeout, import, LLM errors)
- Heuristic retry sequence detection:
  - Multiple executions of same node within 5-minute window
  - Ordered by start time
- Node-level failure statistics
- Retry success rate calculation
- Error distribution across workflows
- Quality risk identification (placeholder for future enhancement)

### Using Python API Directly

You can also use the analysis functions programmatically:

```python
from analyze_traces import (
    load_from_json,
    analyze_latency_distribution,
    identify_bottlenecks,
    verify_parallel_execution,
)

# Load exported trace data
dataset = load_from_json("traces_export.json")

# Analyze latency distribution
latency = analyze_latency_distribution(dataset.workflows)
print(f"p50: {latency.p50_minutes:.1f} min")
print(f"p95: {latency.p95_minutes:.1f} min")
print(f"Outliers above 23 min: {len(latency.outliers_above_23min)}")

# Identify bottlenecks
bottlenecks = identify_bottlenecks(dataset.workflows)
print(f"Primary bottleneck: {bottlenecks.primary_bottleneck}")
for node in bottlenecks.node_performances[:5]:
    print(f"  {node.node_name}: {node.avg_duration_seconds:.1f}s")

# Verify parallel execution
parallel = verify_parallel_execution(dataset.workflows)
print(f"Parallel execution: {parallel.is_parallel}")
print(f"Time savings: {parallel.avg_time_savings_seconds:.1f}s")

# Export to CSV
with open("output/latency.csv", "w") as f:
    f.write(latency.to_csv())
```

### Analysis Output Files

The analysis generates CSV files in the `output/` directory:

1. **`latency_distribution.csv`** - Percentile metrics and outlier analysis
   - p50/p95/p99 latency values
   - Min/max/mean/std_dev statistics
   - Outlier counts (>23 min, <7 min)
   - % within claimed 7-23 minute range

2. **`bottleneck_analysis.csv`** - Node-level performance breakdown
   - Node name, execution count
   - Average/median duration, std deviation
   - % of total workflow time
   - Total time across all workflows

3. **`parallel_execution_analysis.csv`** - Parallel execution evidence
   - Parallel vs sequential workflow counts
   - Average start time deltas
   - Time savings calculation
   - Confidence level assessment

## Validating Exports

After exporting data, use the validation utility to check data quality and statistical validity:

```bash
python validate_export.py traces_export.json
```

The validator provides:
- **Dataset overview**: Workflow counts, hierarchical data status
- **Workflow statistics**: Validator presence, duration stats, unique nodes
- **Statistical validity assessment**: Sample size adequacy for each analysis type
- **Recommendations**: Whether the dataset is ready for analysis

**Example output:**
```
DATASET OVERVIEW
  Total workflows:        384
  Hierarchical data:      Yes
  Workflows with validators: 15

STATISTICAL VALIDITY ASSESSMENT
  Latency Analysis:       EXCELLENT (n >= 100)
  Bottleneck Analysis:    EXCELLENT (n >= 100)
  Parallel Analysis:      WEAK (10 <= n < 20, low confidence)

RECOMMENDATION
  Status: READY FOR COMPREHENSIVE ANALYSIS
```

## Export Output Format

The script generates a JSON file with the following structure:

```json
{
  "export_metadata": {
    "export_timestamp": "2025-11-28T12:00:00Z",
    "project_name": "project-name",
    "total_traces": 150,
    "langsmith_api_version": "0.4.x"
  },
  "traces": [
    {
      "id": "run_id",
      "name": "workflow_name",
      "start_time": "2025-11-28T10:00:00Z",
      "end_time": "2025-11-28T10:15:00Z",
      "duration_seconds": 900,
      "status": "success",
      "inputs": {},
      "outputs": {},
      "error": null,
      "run_type": "chain",
      "child_runs": []
    }
  ]
}
```

## Development

### Running Tests

**Export module tests (33 tests):**
```bash
# Run all export tests
pytest test_export_langsmith_traces.py -v

# Run specific test class
pytest test_export_langsmith_traces.py::TestArgumentParsing -v

# Run with coverage
pytest --cov=export_langsmith_traces test_export_langsmith_traces.py
```

**Analysis module tests (31 tests):**
```bash
# Run all analysis tests
pytest test_analyze_traces.py -v

# Run specific phase tests
pytest test_analyze_traces.py::TestLatencyDistribution -v
pytest test_analyze_traces.py::TestBottleneckIdentification -v
pytest test_analyze_traces.py::TestParallelExecutionVerification -v
pytest test_analyze_traces.py::TestCSVExport -v

# Run with coverage
pytest --cov=analyze_traces test_analyze_traces.py
```

**Cost analysis module tests (20 tests):**
```bash
# Run all cost analysis tests
pytest test_analyze_cost.py -v

# Run specific test classes
pytest test_analyze_cost.py::TestPricingConfig -v
pytest test_analyze_cost.py::TestTokenExtraction -v
pytest test_analyze_cost.py::TestCostCalculation -v

# Run with coverage
pytest --cov=analyze_cost test_analyze_cost.py
```

**Failure analysis module tests (15 tests):**
```bash
# Run all failure analysis tests
pytest test_analyze_failures.py -v

# Run specific test classes
pytest test_analyze_failures.py::TestFailureDetection -v
pytest test_analyze_failures.py::TestRetryDetection -v
pytest test_analyze_failures.py::TestNodeFailureAnalysis -v

# Run with coverage
pytest --cov=analyze_failures test_analyze_failures.py
```

**Run all tests:**
```bash
# Run all 99 tests (33 export + 31 analysis + 20 cost + 15 failure)
pytest -v

# Run with coverage
pytest --cov=. -v
```

### Project Structure

```
export-langsmith-data/
├── .env.example                      # API key configuration template
├── .gitignore                       # Git ignore patterns
├── requirements.txt                 # Python dependencies
├── PLAN.md                          # PDCA implementation plan
├── export-langsmith-requirements.md # Export requirements specification
├── export_langsmith_traces.py       # Data export script
├── test_export_langsmith_traces.py  # Export test suite (33 tests)
├── validate_export.py               # Export validation utility
├── test_validate_export.py          # Validation test suite (7 tests)
├── analyze_traces.py                # Performance analysis module (Phase 3A)
├── test_analyze_traces.py           # Analysis test suite (31 tests)
├── analyze_cost.py                  # Cost analysis module (Phase 3B)
├── test_analyze_cost.py             # Cost analysis test suite (20 tests)
├── analyze_failures.py              # Failure pattern analysis module (Phase 3C)
├── test_analyze_failures.py         # Failure analysis test suite (15 tests)
├── verify_analysis_report.py        # Verification tool for all phases
├── notebooks/
│   └── langsmith_trace_performance_analysis.ipynb  # Interactive analysis notebook
├── output/                          # Generated CSV analysis results
│   ├── latency_distribution.csv
│   ├── bottleneck_analysis.csv
│   └── parallel_execution_analysis.csv
└── README.md                        # This file
```

### Development Approach

This project follows the **PDCA (Plan-Do-Check-Act) framework** with strict Test-Driven Development:
- **Plan**: Comprehensive analysis and implementation plan (see PLAN.md)
- **Do**: TDD implementation with Red-Green-Refactor cycles
- **Check**: Validation against requirements and quality standards
- **Act**: Retrospection and continuous improvement

## Implementation Status

### ✅ Complete - Production Ready

**Data Export Module:**
- ✅ Project setup with virtual environment (uv/venv)
- ✅ Dependencies configuration with CI/CD quality gates
- ✅ CLI argument parsing with validation
- ✅ **Environment variable support** - Optional `.env` file configuration for simplified usage
- ✅ **Automatic pagination** - Handles API 100-record limit with multi-page fetching and progress indication
- ✅ **Hierarchical data export** - `--include-children` flag for complete workflow structures
- ✅ LangSmith client initialization with authentication
- ✅ Run fetching with exponential backoff rate limiting
- ✅ Data formatting and transformation with safe field extraction
- ✅ JSON export functionality with error handling
- ✅ Comprehensive error scenario handling
- ✅ Main orchestration with user-friendly progress feedback
- ✅ End-to-end integration testing
- ✅ Test suite: 33 tests, high coverage
- ✅ Code quality: Black, Ruff, mypy, Bandit, Safety checks passing

**Performance Analysis Module:**
- ✅ Data loading from JSON exports with hierarchical support
- ✅ **Latency distribution analysis** - p50/p95/p99 percentiles, outlier detection
- ✅ **Bottleneck identification** - Node-level performance ranking and metrics
- ✅ **Parallel execution verification** - Detect parallel validators, calculate time savings
- ✅ **CSV export functionality** - Export all analysis results to CSV format
- ✅ **Interactive Jupyter notebook** - Complete analysis workflow with visualizations
- ✅ Test suite: 31 tests (10 Phase 1 + 5 Phase 2 + 6 Phase 3 + 6 Phase 4 + 4 Phase 5)
- ✅ Type-safe implementation with mypy strict mode
- ✅ Code quality: Black, Ruff, mypy checks passing
- ✅ TDD methodology: Strict RED-GREEN-REFACTOR cycles across all 5 phases

### ✅ Complete - Production Ready (Continued)

**Cost Analysis Module (Phase 3B):**
- ✅ Configurable pricing models for any LLM provider
- ✅ Token usage extraction from trace metadata
- ✅ Cost calculation with input/output/cache token pricing
- ✅ Workflow-level cost aggregation
- ✅ Node-level cost breakdown with percentages
- ✅ Scaling projections (1x, 10x, 100x, 1000x)
- ✅ Test suite: 20 tests, full coverage
- ✅ Code quality: Black, Ruff, mypy, Bandit checks passing

**Failure Pattern Analysis Module (Phase 3C):**
- ✅ Status-based failure detection
- ✅ Regex-based error classification (5 patterns + unknown)
- ✅ Heuristic retry sequence detection
- ✅ Node-level failure statistics
- ✅ Retry success rate calculation
- ✅ Error distribution tracking
- ✅ Test suite: 15 tests, full coverage
- ✅ Code quality: Black, Ruff, mypy, Bandit checks passing

### Optional Features Not Implemented

- ⏸️ Progress indication (tqdm) - Skipped in favor of simple console output
- ⏸️ Validator effectiveness analysis - Placeholder in Phase 3C for future enhancement
- ⏸️ Cache effectiveness analysis - Placeholder in Phase 3B for future enhancement

## Troubleshooting

### Authentication Errors
- Verify your API key is correct in `.env` or command line
- Check API key has not expired at https://smith.langchain.com/settings

### Rate Limit Errors
- Script automatically retries with exponential backoff
- Consider reducing export frequency if hitting limits repeatedly

### Network Errors
- Check internet connectivity
- Verify access to https://api.smith.langchain.com
- Check firewall/proxy settings

## License

Property of Stride http://www.stride.build

## Contributing

This project was developed using Claude Code with the PDCA framework. See export-langsmith-implementation-plan.md for implementation details.

## References

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangSmith Python SDK](https://github.com/langchain-ai/langsmith-sdk)
- [PDCA Framework](https://github.com/kenjudy/human-ai-collaboration-process)
