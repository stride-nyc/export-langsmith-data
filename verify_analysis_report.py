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
    dataset: TraceDataset, expected: Optional[Dict[str, Any]] = None
) -> LatencyDistribution:
    """Verify all latency distribution calculations."""
    print_header("LATENCY DISTRIBUTION VERIFICATION")

    latency_dist = analyze_latency_distribution(dataset.workflows)

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
    below_7_count = len(latency_dist.outliers_below_7min)
    above_23_count = len(latency_dist.outliers_above_23min)
    within_7_23 = latency_dist.percent_within_7_23_claim

    print(
        f"Below 7 min:     {below_7_count} workflows ({below_7_count/total_workflows*100:.1f}%)"
    )
    print(f"Within 7-23 min: {within_7_23:.1f}%")
    print(
        f"Above 23 min:    {above_23_count} workflows ({above_23_count/total_workflows*100:.1f}%)"
    )

    if expected and "outliers" in expected:
        print("\nExpected Values Verification:")
        exp = expected["outliers"]
        check_value(
            "% below 7 min",
            exp.get("below_7_pct", 0),
            below_7_count / total_workflows * 100,
        )
        check_value("% within 7-23 min", exp.get("within_7_23_pct", 0), within_7_23)
        check_value(
            "% above 23 min",
            exp.get("above_23_pct", 0),
            above_23_count / total_workflows * 100,
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
    print(f"  Below 7 min: {len(latency_dist.outliers_below_7min)/total*100:.1f}%")
    print(f"  Within 7-23 min: {latency_dist.percent_within_7_23_claim:.1f}%")
    print(f"  Above 23 min: {len(latency_dist.outliers_above_23min)/total*100:.1f}%")

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

    # Run all verifications
    verify_dataset_info(dataset, expected)
    latency_dist = verify_latency_distribution(dataset, expected)
    bottleneck_analysis = verify_bottleneck_analysis(dataset, expected)
    parallel_evidence = verify_parallel_execution(dataset, expected)
    generate_summary_report(
        dataset, latency_dist, bottleneck_analysis, parallel_evidence
    )

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
