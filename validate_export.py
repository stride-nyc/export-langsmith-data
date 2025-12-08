#!/usr/bin/env python3
"""
Validate LangSmith trace export files.

This script validates the structure and content of exported trace data files,
providing statistics on workflows, validators, and data quality.

Usage:
    python validate_export.py <export_file.json>

Example:
    python validate_export.py traces_export.json
"""

import sys
from pathlib import Path
from typing import Set

from analyze_traces import load_from_json


def validate_export_file(filepath: str) -> None:
    """
    Validate an export file and print comprehensive statistics.

    Args:
        filepath: Path to the JSON export file

    Raises:
        FileNotFoundError: If the export file doesn't exist
        Exception: If the file cannot be loaded or is invalid
    """
    # Check file exists
    file_path = Path(filepath)
    if not file_path.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    # Get file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"\nValidating export file: {filepath}")
    print(f"File size: {file_size_mb:.1f} MB")
    print("-" * 70)

    # Load the dataset
    print("\nLoading trace data...")
    try:
        dataset = load_from_json(filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print("Data loaded successfully!")

    # Basic statistics
    print(f"\n{'='*70}")
    print("DATASET OVERVIEW")
    print(f"{'='*70}")
    print(f"  Total workflows:        {len(dataset.workflows)}")
    print(f"  Orphan traces:          {len(dataset.orphan_traces)}")
    print(
        f"  Total traces:           {len(dataset.workflows) + len(dataset.orphan_traces)}"
    )
    print(f"  Hierarchical data:      {'Yes' if dataset.is_hierarchical else 'No'}")

    # Workflow statistics
    if dataset.workflows:
        print(f"\n{'='*70}")
        print("WORKFLOW STATISTICS")
        print(f"{'='*70}")

        # Count workflows with validators
        validator_names = [
            "meta_evaluation",
            "normative_validation",
            "simulated_testing",
        ]
        workflows_with_validators = 0
        workflows_with_all_validators = 0

        for workflow in dataset.workflows:
            has_any_validator = any(
                v_name in workflow.nodes for v_name in validator_names
            )
            has_all_validators = all(
                v_name in workflow.nodes for v_name in validator_names
            )

            if has_any_validator:
                workflows_with_validators += 1
            if has_all_validators:
                workflows_with_all_validators += 1

        print(f"  Workflows with any validator:  {workflows_with_validators}")
        print(f"  Workflows with all validators: {workflows_with_all_validators}")

        # Duration statistics
        durations = [
            w.total_duration / 60 for w in dataset.workflows if w.total_duration > 0
        ]
        if durations:
            print("\n  Duration Statistics (minutes):")
            print(f"    Min:     {min(durations):>8.2f}")
            print(f"    Max:     {max(durations):>8.2f}")
            print(f"    Average: {sum(durations)/len(durations):>8.2f}")
            print(f"    Count:   {len(durations):>8}")

        # Node statistics
        all_node_names: Set[str] = set()
        for workflow in dataset.workflows:
            all_node_names.update(workflow.nodes.keys())

        print(f"\n  Unique node types: {len(all_node_names)}")
        if all_node_names:
            print(f"    Nodes: {', '.join(sorted(all_node_names))}")

    # Statistical validity assessment
    print(f"\n{'='*70}")
    print("STATISTICAL VALIDITY ASSESSMENT")
    print(f"{'='*70}")

    total_workflows = len(dataset.workflows)
    validator_workflows = workflows_with_validators if dataset.workflows else 0

    # Latency & Bottleneck analysis
    if total_workflows >= 100:
        print("  Latency Analysis:       EXCELLENT (n >= 100)")
    elif total_workflows >= 30:
        print("  Latency Analysis:       GOOD (n >= 30)")
    elif total_workflows >= 10:
        print("  Latency Analysis:       ACCEPTABLE (n >= 10)")
    else:
        print("  Latency Analysis:       INSUFFICIENT (n < 10)")

    # Bottleneck analysis
    if total_workflows >= 100:
        print("  Bottleneck Analysis:    EXCELLENT (n >= 100)")
    elif total_workflows >= 30:
        print("  Bottleneck Analysis:    GOOD (n >= 30)")
    elif total_workflows >= 10:
        print("  Bottleneck Analysis:    ACCEPTABLE (n >= 10)")
    else:
        print("  Bottleneck Analysis:    INSUFFICIENT (n < 10)")

    # Parallel execution analysis
    if validator_workflows >= 30:
        print("  Parallel Analysis:      GOOD (n >= 30)")
    elif validator_workflows >= 20:
        print("  Parallel Analysis:      ACCEPTABLE (20 <= n < 30)")
    elif validator_workflows >= 10:
        print("  Parallel Analysis:      WEAK (10 <= n < 20, low confidence)")
    else:
        print("  Parallel Analysis:      INSUFFICIENT (n < 10)")

    # Hierarchical data check
    if not dataset.is_hierarchical:
        print("\n  WARNING: No hierarchical data detected!")
        print("           Re-export with --include-children flag for full analysis.")

    # Final recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    if total_workflows >= 100 and validator_workflows >= 10:
        print("  Status: READY FOR COMPREHENSIVE ANALYSIS")
        print("  - Strong latency and bottleneck analysis")
        if validator_workflows >= 20:
            print("  - Good parallel execution analysis")
        else:
            print("  - Acceptable parallel execution analysis (with caveats)")
    elif total_workflows >= 30:
        print("  Status: ACCEPTABLE FOR BASIC ANALYSIS")
        print("  - Can proceed with analysis")
        print("  - Consider exporting more data for stronger conclusions")
    else:
        print("  Status: INSUFFICIENT DATA")
        print("  - Export more traces for meaningful analysis")
        print(f"  - Recommend at least 100 workflows (current: {total_workflows})")

    print()


def main() -> None:
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python validate_export.py <export_file.json>")
        print("\nExample:")
        print("  python validate_export.py traces_export.json")
        sys.exit(1)

    export_file = sys.argv[1]
    validate_export_file(export_file)


if __name__ == "__main__":
    main()
