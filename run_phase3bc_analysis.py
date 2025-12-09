"""
Run Phase 3B (Cost) and Phase 3C (Failure) analyses on LangSmith trace data.

Generates intermediate data files for the Assessment/data directory.
"""

import json
from pathlib import Path
from analyze_cost import analyze_costs, EXAMPLE_PRICING_CONFIGS
from analyze_failures import analyze_failures
from analyze_traces import load_from_json

# File paths
# Note: Using existing data file - no token usage data available in LangSmith export
# Cost analysis will note this limitation
INPUT_FILE = Path(
    r"C:\Users\Ken Judy\source\repos\neotalogic-ai-demo-214120cfb40b\Assessment\data\neota_agent_traces_complete_workflows.json"
)
OUTPUT_DIR = Path(
    r"C:\Users\Ken Judy\source\repos\neotalogic-ai-demo-214120cfb40b\Assessment\data"
)


def run_cost_analysis(dataset):
    """Run Phase 3B cost analysis."""
    print("\n" + "=" * 80)
    print("PHASE 3B: COST ANALYSIS")
    print("=" * 80)

    # Use Gemini 1.5 Pro pricing
    pricing_config = EXAMPLE_PRICING_CONFIGS["gemini_1.5_pro"]

    print(f"\nPricing Model: {pricing_config.model_name}")
    print(f"  Input tokens: ${pricing_config.input_tokens_per_1k:.6f} per 1K")
    print(f"  Output tokens: ${pricing_config.output_tokens_per_1k:.6f} per 1K")
    print(f"  Cache reads: ${pricing_config.cache_read_per_1k:.6f} per 1K")

    # Run analysis
    results = analyze_costs(
        workflows=dataset.workflows,
        pricing_config=pricing_config,
        scaling_factors=[1, 10, 100, 1000],
        monthly_workflow_estimate=500,  # Estimate based on walkthrough
    )

    # Print summary
    print(f"\n{'Workflow Cost Statistics':=^80}")
    print(f"  Total workflows analyzed: {results.total_workflows_analyzed}")
    print(f"  Average cost per workflow: ${results.avg_cost_per_workflow:.4f}")
    print(f"  Median cost per workflow: ${results.median_cost_per_workflow:.4f}")
    print(f"  Min cost: ${results.min_cost:.4f}")
    print(f"  Max cost: ${results.max_cost:.4f}")

    print(f"\n{'Node Cost Breakdown (Top 5)':=^80}")
    for i, node in enumerate(results.node_summaries[:5], 1):
        print(f"\n{i}. {node.node_name}")
        print(f"   Total cost: ${node.total_cost:.4f}")
        print(f"   Executions: {node.execution_count}")
        print(f"   Avg per execution: ${node.avg_cost_per_execution:.6f}")
        print(f"   % of total: {node.percent_of_total_cost:.1f}%")

    print(f"\n{'Scaling Projections':=^80}")
    for scale_label, projection in results.scaling_projections.items():
        print(f"\n{scale_label} volume:")
        print(f"  Workflow count: {projection.workflow_count:,}")
        print(f"  Total cost: ${projection.total_cost:,.2f}")
        if projection.cost_per_month_30days:
            print(f"  Monthly cost (500 workflows/month): ${projection.cost_per_month_30days:,.2f}/month")

    if results.data_quality_notes:
        print(f"\n{'Data Quality Notes':=^80}")
        for note in results.data_quality_notes:
            print(f"  - {note}")

    # Check if we have any cost data
    if results.avg_cost_per_workflow == 0.0:
        print(f"\n{'WARNING':=^80}")
        print("  No token usage data found in traces!")
        print("  LangSmith may not be configured to record token usage.")
        print("  Cost analysis cannot be performed without token data.")
        print("  ")
        print("  This is a limitation of the data source, not the analysis tools.")
        print("  To enable cost analysis:")
        print("    1. Check LangSmith project settings for token tracking")
        print("    2. Ensure LLM calls include usage_metadata in outputs")
        print("    3. Re-export data after enabling token tracking")

    # Save intermediate data
    output_file = OUTPUT_DIR / "phase3b_cost_analysis_data.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "summary": {
                    "total_workflows": results.total_workflows_analyzed,
                    "avg_cost_per_workflow": results.avg_cost_per_workflow,
                    "median_cost": results.median_cost_per_workflow,
                    "min_cost": results.min_cost,
                    "max_cost": results.max_cost,
                    "top_cost_driver": results.top_cost_driver,
                },
                "node_costs": [
                    {
                        "node_name": n.node_name,
                        "total_cost": n.total_cost,
                        "execution_count": n.execution_count,
                        "avg_cost_per_execution": n.avg_cost_per_execution,
                        "percent_of_total": n.percent_of_total_cost,
                    }
                    for n in results.node_summaries
                ],
                "scaling_projections": {
                    label: {
                        "scale_factor": p.scale_factor,
                        "workflow_count": p.workflow_count,
                        "total_cost": p.total_cost,
                        "monthly_cost": p.cost_per_month_30days,
                    }
                    for label, p in results.scaling_projections.items()
                },
                "data_quality_notes": results.data_quality_notes,
            },
            f,
            indent=2,
        )
    print(f"\n[OK] Saved intermediate data to: {output_file}")

    return results


def run_failure_analysis(dataset):
    """Run Phase 3C failure analysis."""
    print("\n" + "=" * 80)
    print("PHASE 3C: FAILURE PATTERN ANALYSIS")
    print("=" * 80)

    # Run analysis
    results = analyze_failures(workflows=dataset.workflows)

    # Print summary
    print(f"\n{'Overall Metrics':=^80}")
    print(f"  Total workflows: {results.total_workflows}")
    print(f"  Successful workflows: {results.successful_workflows}")
    print(f"  Failed workflows: {results.failed_workflows}")
    print(f"  Overall success rate: {results.overall_success_rate_percent:.1f}%")

    print(f"\n{'Node Failure Breakdown (Top 10)':=^80}")
    for i, node in enumerate(results.node_failure_stats[:10], 1):
        print(f"\n{i}. {node.node_name}")
        print(f"   Failure rate: {node.failure_rate_percent:.1f}%")
        print(f"   Failures: {node.failure_count}/{node.total_executions}")
        print(f"   Success: {node.success_count}")
        print(f"   Retry sequences: {node.retry_sequences_detected}")
        if node.common_error_types:
            print(f"   Common errors: {node.common_error_types}")

    print(f"\n{'Error Distribution':=^80}")
    for error_type, count in sorted(
        results.error_type_distribution.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {error_type}: {count}")

    print(f"\n{'Retry Analysis':=^80}")
    print(f"  Total retry sequences: {results.total_retry_sequences}")
    if results.retry_success_rate_percent is not None:
        print(f"  Retry success rate: {results.retry_success_rate_percent:.1f}%")

    if results.retry_sequences:
        print(f"\n  Sample retry sequences (first 5):")
        for i, seq in enumerate(results.retry_sequences[:5], 1):
            print(f"\n  {i}. {seq.node_name}")
            print(f"     Attempts: {seq.attempt_count}")
            print(f"     Final status: {seq.final_status}")
            print(f"     Total duration: {seq.total_duration_seconds:.1f}s")

    # Save intermediate data
    output_file = OUTPUT_DIR / "phase3c_failure_analysis_data.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "summary": {
                    "total_workflows": results.total_workflows,
                    "successful_workflows": results.successful_workflows,
                    "failed_workflows": results.failed_workflows,
                    "success_rate_percent": results.overall_success_rate_percent,
                    "highest_failure_node": results.highest_failure_node,
                    "most_common_error_type": results.most_common_error_type,
                },
                "node_failures": [
                    {
                        "node_name": n.node_name,
                        "total_executions": n.total_executions,
                        "failure_count": n.failure_count,
                        "success_count": n.success_count,
                        "failure_rate_percent": n.failure_rate_percent,
                        "retry_sequences": n.retry_sequences_detected,
                        "avg_retries": n.avg_retries_when_failing,
                        "common_errors": n.common_error_types,
                    }
                    for n in results.node_failure_stats
                ],
                "error_distribution": results.error_type_distribution,
                "retry_analysis": {
                    "total_sequences": results.total_retry_sequences,
                    "success_rate_percent": results.retry_success_rate_percent,
                    "sample_sequences": [
                        {
                            "node_name": s.node_name,
                            "attempts": s.attempt_count,
                            "final_status": s.final_status,
                            "total_duration": s.total_duration_seconds,
                        }
                        for s in results.retry_sequences[:10]
                    ],
                },
            },
            f,
            indent=2,
        )
    print(f"\n[OK] Saved intermediate data to: {output_file}")

    return results


def main():
    """Main execution function."""
    print(f"\nLoading data from: {INPUT_FILE}")
    dataset = load_from_json(str(INPUT_FILE))

    print(f"Loaded {len(dataset.workflows)} workflows")
    print(f"Hierarchical data: {dataset.is_hierarchical}")

    # Run Phase 3B: Cost Analysis
    cost_results = run_cost_analysis(dataset)

    # Run Phase 3C: Failure Analysis
    failure_results = run_failure_analysis(dataset)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nIntermediate data files saved to:")
    print(f"  - {OUTPUT_DIR / 'phase3b_cost_analysis_data.json'}")
    print(f"  - {OUTPUT_DIR / 'phase3c_failure_analysis_data.json'}")
    print("\nReady to generate Phase 3B and 3C reports.")


if __name__ == "__main__":
    main()
