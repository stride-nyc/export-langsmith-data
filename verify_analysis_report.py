"""
Verification tool for LangSmith trace analysis reports.

This script regenerates all statistics and calculations from an analysis
to provide deterministic verification of findings.

Usage:
    python verify_analysis_report.py <input_file.json> [--expected-values expected.json]

Examples:
    # Basic verification
    python verify_analysis_report.py traces.json

    # Verify against expected values
    python verify_analysis_report.py traces.json --expected-values expected_stats.json
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from analyze_traces import (
    load_from_json,
    analyze_latency_distribution,
    identify_bottlenecks,
    verify_parallel_execution as analyze_parallel_execution,
    TraceDataset,
    LatencyDistribution,
    BottleneckAnalysis,
    ParallelExecutionEvidence,
)
from analyze_cost import (
    analyze_costs,
    PricingConfig,
    EXAMPLE_PRICING_CONFIGS,
    CostAnalysisResults,
)
from analyze_failures import (
    analyze_failures,
    FailureAnalysisResults,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_section(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * 80)


def check_value(
    description: str, expected: float, actual: float, tolerance: float = 0.1
) -> bool:
    """
    Check if actual value matches expected within tolerance.

    Args:
        description: Description of the value being checked
        expected: Expected value
        actual: Actual value from analysis
        tolerance: Acceptable difference

    Returns:
        True if values match within tolerance
    """
    match = abs(actual - expected) < tolerance
    status = "PASS" if match else "FAIL"
    print(
        f"  {description:<40} Expected: {expected:>10.2f}  Actual: {actual:>10.2f}  [{status}]"
    )
    return match


def verify_dataset_info(
    dataset: TraceDataset, expected: Optional[Dict[str, Any]] = None
) -> int:
    """Verify basic dataset information."""
    print_header("DATASET INFORMATION")

    sample_size = len(dataset.workflows)
    print(f"Sample Size: {sample_size} complete workflows")
    print(f"Hierarchical Data: {dataset.is_hierarchical}")

    if expected:
        exp_size = expected.get("sample_size")
        if exp_size is not None:
            print(f"\nExpected sample size: {exp_size}")
            print(f"Match: {sample_size == exp_size}")

    return sample_size


def verify_latency_distribution(
    dataset: TraceDataset,
    expected: Optional[Dict[str, Any]] = None,
    min_threshold: float = 7.0,
    max_threshold: float = 40.0,
) -> LatencyDistribution:
    """Verify all latency distribution calculations."""
    print_header("LATENCY DISTRIBUTION VERIFICATION")

    latency_dist = analyze_latency_distribution(
        dataset.workflows, min_threshold=min_threshold, max_threshold=max_threshold
    )

    # Percentile Metrics
    print_section("Percentile Metrics")
    print(f"p50 (median):    {latency_dist.p50_minutes:.2f} minutes")
    print(f"p95:             {latency_dist.p95_minutes:.2f} minutes")
    print(f"p99:             {latency_dist.p99_minutes:.2f} minutes")
    print(f"Min:             {latency_dist.min_minutes:.2f} minutes")
    print(f"Max:             {latency_dist.max_minutes:.2f} minutes")
    print(f"Mean:            {latency_dist.mean_minutes:.2f} minutes")
    print(f"Std Dev:         {latency_dist.std_dev_minutes:.2f} minutes")

    # Verify against expected values if provided
    if expected and "latency" in expected:
        print("\nExpected Values Verification:")
        exp = expected["latency"]
        check_value("p50", exp.get("p50", 0), latency_dist.p50_minutes)
        check_value("p95", exp.get("p95", 0), latency_dist.p95_minutes)
        check_value("p99", exp.get("p99", 0), latency_dist.p99_minutes)
        check_value("min", exp.get("min", 0), latency_dist.min_minutes)
        check_value("max", exp.get("max", 0), latency_dist.max_minutes)
        check_value("mean", exp.get("mean", 0), latency_dist.mean_minutes)

    # Outlier Analysis
    print_section("Outlier Analysis")
    total_workflows = len(dataset.workflows)
    below_min_count = len(latency_dist.outliers_below_min)
    above_max_count = len(latency_dist.outliers_above_max)
    within_range = latency_dist.percent_within_range

    print(
        f"Below {min_threshold} min:     {below_min_count} workflows ({below_min_count/total_workflows*100:.1f}%)"
    )
    print(f"Within {min_threshold}-{max_threshold} min: {within_range:.1f}%")
    print(
        f"Above {max_threshold} min:    {above_max_count} workflows ({above_max_count/total_workflows*100:.1f}%)"
    )

    if expected and "outliers" in expected:
        print("\nExpected Values Verification:")
        exp = expected["outliers"]
        check_value(
            f"% below {min_threshold} min",
            exp.get("below_min_pct", 0),
            below_min_count / total_workflows * 100,
        )
        check_value(
            f"% within {min_threshold}-{max_threshold} min",
            exp.get("within_range_pct", 0),
            within_range,
        )
        check_value(
            f"% above {max_threshold} min",
            exp.get("above_max_pct", 0),
            above_max_count / total_workflows * 100,
        )

    return latency_dist


def verify_bottleneck_analysis(
    dataset: TraceDataset, expected: Optional[Dict[str, Any]] = None
) -> BottleneckAnalysis:
    """Verify bottleneck identification calculations."""
    print_header("BOTTLENECK IDENTIFICATION VERIFICATION")

    bottleneck_analysis = identify_bottlenecks(dataset.workflows)

    print_section("Primary Bottleneck")
    print(f"Primary: {bottleneck_analysis.primary_bottleneck}")
    print(f"Top 3:   {', '.join(bottleneck_analysis.top_3_bottlenecks)}")

    if expected and "bottleneck" in expected:
        exp = expected["bottleneck"]
        print("\nExpected Values Verification:")
        expected_primary = exp.get("primary")
        if expected_primary:
            match = bottleneck_analysis.primary_bottleneck == expected_primary
            print(
                f"  Primary bottleneck = {expected_primary}: [{'PASS' if match else 'FAIL'}]"
            )

    # Detailed node performance
    print_section("Top 10 Node Performance")
    print(
        f"{'Rank':<5} {'Node Name':<35} {'Count':<7} {'Avg Dur':<10} {'% Workflow':<12}"
    )
    print("-" * 80)

    for i, node in enumerate(bottleneck_analysis.node_performances[:10], 1):
        print(
            f"{i:<5} {node.node_name:<35} {node.execution_count:<7} "
            f"{node.avg_duration_seconds:>8.1f}s {node.avg_percent_of_workflow:>10.1f}%"
        )

    return bottleneck_analysis


def verify_parallel_execution(
    dataset: TraceDataset, expected: Optional[Dict[str, Any]] = None
) -> ParallelExecutionEvidence:
    """Verify parallel execution detection and calculations."""
    print_header("PARALLEL EXECUTION VERIFICATION")

    parallel_evidence = analyze_parallel_execution(dataset.workflows)

    print_section("Detection Method")
    print("Heuristic: Validators starting within 5 seconds = PARALLEL")
    print("Data Source: Direct observation of start_time timestamps from LangSmith")

    print_section("Workflow Classification")
    total = len(dataset.workflows)
    parallel_count = parallel_evidence.parallel_confirmed_count
    sequential_count = parallel_evidence.sequential_count

    print(
        f"Parallel workflows:   {parallel_count}/{total} ({parallel_count/total*100:.1f}%)"
    )
    print(
        f"Sequential workflows: {sequential_count}/{total} ({sequential_count/total*100:.1f}%)"
    )

    if expected and "parallel" in expected:
        exp = expected["parallel"]
        print("\nExpected Values Verification:")
        check_value(
            "% parallel", exp.get("parallel_pct", 0), parallel_count / total * 100
        )
        check_value(
            "% sequential", exp.get("sequential_pct", 0), sequential_count / total * 100
        )

    print_section("Timing Metrics")
    print(
        f"Avg start time delta:  {parallel_evidence.avg_start_time_delta_seconds:.1f}s ({parallel_evidence.avg_start_time_delta_seconds/60:.1f} min)"
    )
    print(
        f"Avg sequential time:   {parallel_evidence.avg_sequential_time_seconds:.1f}s ({parallel_evidence.avg_sequential_time_seconds/60:.1f} min)"
    )
    print(
        f"Avg parallel time:     {parallel_evidence.avg_parallel_time_seconds:.1f}s ({parallel_evidence.avg_parallel_time_seconds/60:.1f} min)"
    )
    print(
        f"Avg time savings:      {parallel_evidence.avg_time_savings_seconds:.1f}s ({parallel_evidence.avg_time_savings_seconds/60:.1f} min)"
    )

    if expected and "parallel" in expected:
        exp = expected["parallel"]
        print("\nExpected Values Verification:")
        check_value(
            "Start delta (s)",
            exp.get("start_delta_s", 0),
            parallel_evidence.avg_start_time_delta_seconds,
        )
        check_value(
            "Sequential time (s)",
            exp.get("sequential_s", 0),
            parallel_evidence.avg_sequential_time_seconds,
        )
        check_value(
            "Parallel time (s)",
            exp.get("parallel_s", 0),
            parallel_evidence.avg_parallel_time_seconds,
        )
        check_value(
            "Time savings (s)",
            exp.get("savings_s", 0),
            parallel_evidence.avg_time_savings_seconds,
        )

    return parallel_evidence


def generate_summary_report(
    dataset: TraceDataset,
    latency_dist: LatencyDistribution,
    bottleneck_analysis: BottleneckAnalysis,
    parallel_evidence: ParallelExecutionEvidence,
) -> None:
    """Generate final summary with all key findings."""
    print_header("ANALYSIS SUMMARY")

    total = len(dataset.workflows)

    print("\nLatency Distribution:")
    print(f"  p50 (median): {latency_dist.p50_minutes:.2f} min")
    print(f"  p95: {latency_dist.p95_minutes:.2f} min")
    print(f"  p99: {latency_dist.p99_minutes:.2f} min")
    print(f"  Mean: {latency_dist.mean_minutes:.2f} min")
    print(
        f"  Range: {latency_dist.min_minutes:.2f} - {latency_dist.max_minutes:.2f} min"
    )

    print("\nOutliers:")
    print(
        f"  Below {latency_dist.min_threshold} min: {len(latency_dist.outliers_below_min)/total*100:.1f}%"
    )
    print(
        f"  Within {latency_dist.min_threshold}-{latency_dist.max_threshold} min: {latency_dist.percent_within_range:.1f}%"
    )
    print(
        f"  Above {latency_dist.max_threshold} min: {len(latency_dist.outliers_above_max)/total*100:.1f}%"
    )

    print("\nBottlenecks:")
    print(f"  Primary: {bottleneck_analysis.primary_bottleneck}")
    print(f"  Top 3: {', '.join(bottleneck_analysis.top_3_bottlenecks)}")

    print("\nParallel Execution:")
    print(
        f"  Parallel: {parallel_evidence.parallel_confirmed_count}/{total} ({parallel_evidence.parallel_confirmed_count/total*100:.1f}%)"
    )
    print(
        f"  Sequential: {parallel_evidence.sequential_count}/{total} ({parallel_evidence.sequential_count/total*100:.1f}%)"
    )
    print(
        f"  Time savings if parallel: {parallel_evidence.avg_time_savings_seconds/60:.1f} min"
    )


def verify_cost_analysis(
    dataset: TraceDataset,
    pricing_config: PricingConfig,
    expected: Optional[Dict[str, Any]] = None,
) -> CostAnalysisResults:
    """
    Verify Phase 3B cost calculations.

    Displays:
    - Cost per workflow (avg, median, range)
    - Top 3 cost drivers
    - Scaling projections (10x, 100x, 1000x)
    - Cache effectiveness if available
    """
    print_header("PHASE 3B: COST ANALYSIS VERIFICATION")

    print(f"\nPricing Model: {pricing_config.model_name}")
    print(f"  Input tokens: ${pricing_config.input_tokens_per_1k}/1K tokens")
    print(f"  Output tokens: ${pricing_config.output_tokens_per_1k}/1K tokens")
    if pricing_config.cache_read_per_1k:
        print(f"  Cache read tokens: ${pricing_config.cache_read_per_1k}/1K tokens")

    # Run cost analysis
    results = analyze_costs(dataset.workflows, pricing_config)

    print_section("Workflow Cost Statistics")
    print(f"  Total workflows analyzed: {results.total_workflows_analyzed}")
    print(f"  Average cost per workflow: ${results.avg_cost_per_workflow:.4f}")
    print(f"  Median cost per workflow: ${results.median_cost_per_workflow:.4f}")
    print(f"  Cost range: ${results.min_cost:.4f} - ${results.max_cost:.4f}")

    if expected and "cost_analysis" in expected:
        exp_cost = expected["cost_analysis"]
        check_value(
            "avg_cost_per_workflow",
            results.avg_cost_per_workflow,
            exp_cost.get("avg_cost_per_workflow"),
        )

    print_section("Cost Drivers (Top 3 Nodes)")
    for i, node in enumerate(results.node_summaries[:3], 1):
        print(
            f"  {i}. {node.node_name}: ${node.total_cost:.4f} ({node.percent_of_total_cost:.1f}%)"
        )
        print(
            f"     Executions: {node.execution_count}, Avg: ${node.avg_cost_per_execution:.4f}"
        )

    print_section("Scaling Projections")
    for scale_label in ["1x", "10x", "100x", "1000x"]:
        if scale_label in results.scaling_projections:
            proj = results.scaling_projections[scale_label]
            print(
                f"  {scale_label}: {proj.workflow_count} workflows → ${proj.total_cost:.2f}"
            )
            if proj.cost_per_month_30days:
                print(f"       Monthly (30 days): ${proj.cost_per_month_30days:.2f}")

    if results.cache_effectiveness_percent:
        print_section("Cache Effectiveness")
        print(f"  Cache hit rate: {results.cache_effectiveness_percent:.1f}%")
        if results.cache_savings_dollars:
            print(f"  Cost savings: ${results.cache_savings_dollars:.2f}")

    if results.data_quality_notes:
        print_section("Data Quality Notes")
        for note in results.data_quality_notes:
            print(f"  - {note}")

    return results


def verify_failure_analysis(
    dataset: TraceDataset, expected: Optional[Dict[str, Any]] = None
) -> FailureAnalysisResults:
    """
    Verify Phase 3C failure calculations.

    Displays:
    - Overall success rate
    - Top 5 nodes by failure rate
    - Retry analysis (sequences detected, success rate)
    - Validator effectiveness
    """
    print_header("PHASE 3C: FAILURE PATTERN ANALYSIS VERIFICATION")

    # Run failure analysis
    results = analyze_failures(dataset.workflows)

    print_section("Overall Success/Failure Metrics")
    print(f"  Total workflows: {results.total_workflows}")
    print(f"  Successful: {results.successful_workflows}")
    print(f"  Failed: {results.failed_workflows}")
    print(f"  Success rate: {results.overall_success_rate_percent:.1f}%")

    if expected and "failure_analysis" in expected:
        exp_fail = expected["failure_analysis"]
        check_value(
            "success_rate",
            results.overall_success_rate_percent,
            exp_fail.get("success_rate"),
        )

    print_section("Node Failure Rates (Top 5)")
    for i, node in enumerate(results.node_failure_stats[:5], 1):
        print(f"  {i}. {node.node_name}: {node.failure_rate_percent:.1f}% failure rate")
        print(f"     {node.failure_count}/{node.total_executions} executions failed")
        if node.retry_sequences_detected > 0:
            print(f"     Retry sequences detected: {node.retry_sequences_detected}")

    print_section("Error Distribution")
    if results.error_type_distribution:
        sorted_errors = sorted(
            results.error_type_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for error_type, count in sorted_errors[:5]:
            print(f"  - {error_type}: {count} occurrences")
    else:
        print("  No errors detected")

    print_section("Retry Analysis")
    print(f"  Total retry sequences: {results.total_retry_sequences}")
    if results.retry_success_rate_percent is not None:
        print(f"  Retry success rate: {results.retry_success_rate_percent:.1f}%")
    if results.avg_cost_of_retries is not None:
        print(f"  Avg cost of retries: ${results.avg_cost_of_retries:.4f}")

    if results.validator_analyses:
        print_section("Validator Effectiveness")
        for validator in results.validator_analyses:
            print(f"  {validator.validator_name}:")
            print(f"    Executions: {validator.total_executions}")
            print(f"    Pass rate: {validator.pass_rate_percent:.1f}%")
            print(f"    Necessary: {'Yes' if validator.is_necessary else 'No'}")

    if results.redundant_validators:
        print(
            f"\n  Potentially redundant validators: {', '.join(results.redundant_validators)}"
        )

    if results.quality_risks_at_scale:
        print_section("Quality Risks at Scale")
        for risk in results.quality_risks_at_scale:
            print(f"  - {risk}")

    return results


def main() -> int:
    """Main verification script."""
    parser = argparse.ArgumentParser(
        description="Verify LangSmith trace analysis calculations"
    )
    parser.add_argument(
        "input_file", type=str, help="Path to JSON export file with trace data"
    )
    parser.add_argument(
        "--expected-values",
        type=str,
        help="Optional JSON file with expected values for verification",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="3a",
        help="Phases to verify: 3a, 3b, 3c, or all (default: 3a)",
    )
    parser.add_argument(
        "--pricing-model",
        type=str,
        default="gemini_1.5_pro",
        help="Pricing model for cost analysis (default: gemini_1.5_pro)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=7.0,
        help="Minimum duration threshold in minutes (default: 7.0)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=40.0,
        help="Maximum duration threshold in minutes (default: 40.0)",
    )

    args = parser.parse_args()

    # Load expected values if provided
    expected = None
    if args.expected_values:
        with open(args.expected_values, "r") as f:
            expected = json.load(f)

    print("=" * 80)
    print(" LANGSMITH TRACE ANALYSIS - STATISTICS VERIFICATION")
    print(" Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    # Load data
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"\nError: Input file not found: {input_path}")
        return 1

    print(f"\nLoading data from: {input_path}")
    dataset = load_from_json(str(input_path))

    phases = args.phases.lower()
    run_3a = phases in ["3a", "all"]
    run_3b = phases in ["3b", "all"]
    run_3c = phases in ["3c", "all"]

    # Run Phase 3A verifications
    if run_3a:
        verify_dataset_info(dataset, expected)
        latency_dist = verify_latency_distribution(
            dataset, expected, args.min_duration, args.max_duration
        )
        bottleneck_analysis = verify_bottleneck_analysis(dataset, expected)
        parallel_evidence = verify_parallel_execution(dataset, expected)
        generate_summary_report(
            dataset, latency_dist, bottleneck_analysis, parallel_evidence
        )

    # Run Phase 3B verification (Cost Analysis)
    if run_3b:
        if args.pricing_model in EXAMPLE_PRICING_CONFIGS:
            pricing_config = EXAMPLE_PRICING_CONFIGS[args.pricing_model]
        else:
            print(
                f"\nWarning: Unknown pricing model '{args.pricing_model}', using gemini_1.5_pro"
            )
            pricing_config = EXAMPLE_PRICING_CONFIGS["gemini_1.5_pro"]

        verify_cost_analysis(dataset, pricing_config, expected)

    # Run Phase 3C verification (Failure Analysis)
    if run_3c:
        verify_failure_analysis(dataset, expected)

    # Final status
    print_header("VERIFICATION COMPLETE")
    print("\nAll calculations have been regenerated from the trace data.")
    if expected:
        print(
            "Review PASS/FAIL indicators above for discrepancies against expected values."
        )
    print("\nNote: Minor differences (<0.1) due to rounding are acceptable.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
