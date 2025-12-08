"""
LangSmith Trace Analysis Tool - Phase 3A: Performance Analysis

This module provides data analysis capabilities for LangSmith trace exports,
focusing on performance metrics, bottleneck identification, and parallel
execution verification.

Following PDCA (Plan-Do-Check-Act) methodology with TDD approach.

Author: Generated with Claude Code (PDCA Framework)
Date: 2025-12-08
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
from pathlib import Path
import numpy as np


@dataclass
class Trace:
    """
    Represents a single LangSmith trace/run.

    Attributes:
        id: Unique identifier for the trace
        name: Name of the trace (e.g., 'LangGraph', 'generate_spec')
        start_time: When the trace started execution
        end_time: When the trace completed
        duration_seconds: Total execution time in seconds
        status: Execution status ('success', 'error', etc.)
        run_type: Type of run ('chain', 'llm', 'tool')
        parent_id: ID of parent trace (None for root traces)
        child_ids: List of child trace IDs
        inputs: Input parameters to the trace
        outputs: Output results from the trace
        error: Error message if execution failed
    """

    id: str
    name: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: float
    status: str
    run_type: str
    parent_id: Optional[str]
    child_ids: List[str]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    error: Optional[str]


@dataclass
class Workflow:
    """
    Represents a complete workflow execution with hierarchical structure.

    A workflow typically represents a LangGraph execution with multiple
    child nodes (e.g., generate_spec, validators, xml_transformation).

    Attributes:
        root_trace: The root/parent trace (usually LangGraph)
        nodes: Dictionary mapping node names to their trace executions
        all_traces: Flat list of all traces in the workflow
    """

    root_trace: Trace
    nodes: Dict[str, List[Trace]]
    all_traces: List[Trace]

    @property
    def total_duration(self) -> float:
        """Return the total workflow duration from root trace."""
        return self.root_trace.duration_seconds


@dataclass
class TraceDataset:
    """
    Container for all loaded trace data.

    Attributes:
        workflows: List of complete workflow executions
        orphan_traces: Traces without parent relationships (flat data)
        metadata: Export metadata (timestamp, version, etc.)
        is_hierarchical: True if data has parent-child relationships
    """

    workflows: List[Workflow]
    orphan_traces: List[Trace]
    metadata: Dict[str, Any]
    is_hierarchical: bool


def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO 8601 datetime string to datetime object.

    Args:
        dt_str: ISO 8601 formatted datetime string

    Returns:
        datetime object or None if string is None/empty
    """
    if not dt_str:
        return None
    return datetime.fromisoformat(dt_str)


def _build_trace_from_dict(
    trace_dict: Dict[str, Any], parent_id: Optional[str] = None
) -> Trace:
    """
    Build a Trace object from a dictionary (recursively handles children).

    Args:
        trace_dict: Dictionary containing trace data
        parent_id: ID of parent trace (for child traces)

    Returns:
        Trace object
    """
    # Extract child IDs from child_runs
    child_runs = trace_dict.get("child_runs", [])
    child_ids = [child["id"] for child in child_runs if isinstance(child, dict)]

    trace = Trace(
        id=trace_dict["id"],
        name=trace_dict["name"],
        start_time=_parse_datetime(trace_dict.get("start_time")),
        end_time=_parse_datetime(trace_dict.get("end_time")),
        duration_seconds=trace_dict.get("duration_seconds", 0.0),
        status=trace_dict.get("status", "unknown"),
        run_type=trace_dict.get("run_type", "chain"),
        parent_id=parent_id,
        child_ids=child_ids,
        inputs=trace_dict.get("inputs", {}),
        outputs=trace_dict.get("outputs", {}),
        error=trace_dict.get("error"),
    )

    return trace


def _build_workflow_from_trace(
    root_dict: Dict[str, Any], all_traces: List[Trace]
) -> Workflow:
    """
    Build a Workflow from a root trace dictionary.

    Args:
        root_dict: Dictionary containing root trace data
        all_traces: List of all traces in the workflow

    Returns:
        Workflow object
    """
    # Find root trace in all_traces
    root_trace = next(t for t in all_traces if t.id == root_dict["id"])

    # Group traces by node name (excluding root)
    nodes: Dict[str, List[Trace]] = {}
    for trace in all_traces:
        if trace.id != root_trace.id:
            if trace.name not in nodes:
                nodes[trace.name] = []
            nodes[trace.name].append(trace)

    return Workflow(root_trace=root_trace, nodes=nodes, all_traces=all_traces)


def _flatten_traces(
    trace_dict: Dict[str, Any], parent_id: Optional[str] = None
) -> List[Trace]:
    """
    Recursively flatten a trace dictionary into a list of Trace objects.

    Args:
        trace_dict: Dictionary containing trace data with potential child_runs
        parent_id: ID of parent trace

    Returns:
        List of Trace objects (parent and all descendants)
    """
    traces = []

    # Build current trace
    current_trace = _build_trace_from_dict(trace_dict, parent_id)
    traces.append(current_trace)

    # Recursively process children
    child_runs = trace_dict.get("child_runs", [])
    for child_dict in child_runs:
        if isinstance(child_dict, dict):
            child_traces = _flatten_traces(child_dict, parent_id=current_trace.id)
            traces.extend(child_traces)

    return traces


def load_from_json(filepath: str) -> TraceDataset:
    """
    Load trace data from a JSON export file.

    Supports both hierarchical data (with child_runs) and flat data
    (without parent-child relationships).

    Args:
        filepath: Path to the JSON export file

    Returns:
        TraceDataset containing loaded workflows and/or orphan traces

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    # Load JSON file
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract metadata
    metadata = data.get("export_metadata", {})
    traces_data = data.get("traces", [])

    workflows = []
    orphan_traces = []
    is_hierarchical = False

    # Process each top-level trace
    for trace_dict in traces_data:
        # Flatten the trace and its children
        all_traces = _flatten_traces(trace_dict)

        # Check if this is hierarchical data (has children)
        if len(all_traces) > 1:
            is_hierarchical = True
            # Build workflow
            workflow = _build_workflow_from_trace(trace_dict, all_traces)
            workflows.append(workflow)
        else:
            # Single trace without children - treat as orphan
            orphan_traces.append(all_traces[0])

    return TraceDataset(
        workflows=workflows,
        orphan_traces=orphan_traces,
        metadata=metadata,
        is_hierarchical=is_hierarchical,
    )


# ============================================================================
# Phase 2: Latency Distribution Analysis
# ============================================================================


@dataclass
class LatencyDistribution:
    """
    Results of latency distribution analysis for workflows.

    Attributes:
        p50_minutes: 50th percentile (median) latency in minutes
        p95_minutes: 95th percentile latency in minutes
        p99_minutes: 99th percentile latency in minutes
        min_minutes: Minimum workflow duration in minutes
        max_minutes: Maximum workflow duration in minutes
        mean_minutes: Average workflow duration in minutes
        std_dev_minutes: Standard deviation of workflow durations
        outliers_above_23min: List of workflow IDs above 23 minutes
        outliers_below_7min: List of workflow IDs below 7 minutes
        percent_within_7_23_claim: Percentage of workflows within 7-23 min range
    """

    p50_minutes: float
    p95_minutes: float
    p99_minutes: float
    min_minutes: float
    max_minutes: float
    mean_minutes: float
    std_dev_minutes: float
    outliers_above_23min: List[str]
    outliers_below_7min: List[str]
    percent_within_7_23_claim: float

    def to_csv(self) -> str:
        """
        Export latency distribution results to CSV format.

        Returns:
            CSV-formatted string with metric names and values
        """
        lines = ["metric,value_minutes,outlier_count,notes"]
        lines.append(f"p50,{self.p50_minutes},,50th percentile (median)")
        lines.append(f"p95,{self.p95_minutes},,95th percentile")
        lines.append(f"p99,{self.p99_minutes},,99th percentile")
        lines.append(f"min,{self.min_minutes},,minimum duration")
        lines.append(f"max,{self.max_minutes},,maximum duration")
        lines.append(f"mean,{self.mean_minutes},,average duration")
        lines.append(f"std_dev,{self.std_dev_minutes},,standard deviation")
        lines.append(
            f"outliers_above_23min,,{len(self.outliers_above_23min)},workflows > 23 minutes"
        )
        lines.append(
            f"outliers_below_7min,,{len(self.outliers_below_7min)},workflows < 7 minutes"
        )
        lines.append(
            f"percent_within_7_23_claim,{self.percent_within_7_23_claim},,% within claimed 7-23 min range"
        )
        return "\n".join(lines)


def analyze_latency_distribution(workflows: List[Workflow]) -> LatencyDistribution:
    """
    Analyze latency distribution across workflows.

    Calculates percentiles, identifies outliers, and validates the
    claimed "7-23 minutes" workflow execution timeframe.

    Args:
        workflows: List of Workflow objects to analyze

    Returns:
        LatencyDistribution with calculated metrics

    Raises:
        ValueError: If no valid workflows are provided
    """
    # Filter out zero-duration workflows (incomplete/errored)
    valid_workflows = [w for w in workflows if w.total_duration > 0]

    if not valid_workflows:
        raise ValueError("No valid workflows to analyze (all have zero duration)")

    # Extract durations in minutes
    durations_minutes = np.array([w.total_duration / 60.0 for w in valid_workflows])

    # Calculate percentiles
    p50 = float(np.percentile(durations_minutes, 50))
    p95 = float(np.percentile(durations_minutes, 95))
    p99 = float(np.percentile(durations_minutes, 99))

    # Calculate basic statistics
    min_val = float(np.min(durations_minutes))
    max_val = float(np.max(durations_minutes))
    mean_val = float(np.mean(durations_minutes))
    std_dev = float(np.std(durations_minutes))

    # Identify outliers
    outliers_above = []
    outliers_below = []
    within_claim = 0

    for workflow, duration_min in zip(valid_workflows, durations_minutes):
        if duration_min > 23.0:
            outliers_above.append(workflow.root_trace.id)
        elif duration_min < 7.0:
            outliers_below.append(workflow.root_trace.id)
        else:
            within_claim += 1

    # Calculate percentage within claimed range
    percent_within = (within_claim / len(valid_workflows)) * 100.0

    return LatencyDistribution(
        p50_minutes=p50,
        p95_minutes=p95,
        p99_minutes=p99,
        min_minutes=min_val,
        max_minutes=max_val,
        mean_minutes=mean_val,
        std_dev_minutes=std_dev,
        outliers_above_23min=outliers_above,
        outliers_below_7min=outliers_below,
        percent_within_7_23_claim=percent_within,
    )


# ============================================================================
# Phase 3: Bottleneck Identification
# ============================================================================


@dataclass
class NodePerformance:
    """
    Performance metrics for a single node type across workflows.

    Attributes:
        node_name: Name of the node (e.g., 'generate_spec', 'xml_transformation')
        execution_count: Number of times this node executed across all workflows
        avg_duration_seconds: Average execution time in seconds
        median_duration_seconds: Median execution time in seconds
        std_dev_seconds: Standard deviation of execution times
        avg_percent_of_workflow: Average percentage of total workflow time
        total_time_seconds: Total time spent in this node across all workflows
    """

    node_name: str
    execution_count: int
    avg_duration_seconds: float
    median_duration_seconds: float
    std_dev_seconds: float
    avg_percent_of_workflow: float
    total_time_seconds: float


@dataclass
class BottleneckAnalysis:
    """
    Results of bottleneck identification analysis.

    Attributes:
        node_performances: List of NodePerformance objects (sorted by avg_duration)
        primary_bottleneck: Name of the slowest node (highest avg duration)
        top_3_bottlenecks: List of top 3 slowest node names
    """

    node_performances: List[NodePerformance]
    primary_bottleneck: Optional[str]
    top_3_bottlenecks: List[str]

    def to_csv(self) -> str:
        """
        Export bottleneck analysis results to CSV format.

        Returns:
            CSV-formatted string with node performance metrics
        """
        lines = [
            "node_name,execution_count,avg_duration_seconds,median_duration_seconds,"
            "std_dev_seconds,avg_percent_of_workflow,total_time_seconds"
        ]

        for node_perf in self.node_performances:
            lines.append(
                f"{node_perf.node_name},{node_perf.execution_count},"
                f"{node_perf.avg_duration_seconds},{node_perf.median_duration_seconds},"
                f"{node_perf.std_dev_seconds},{node_perf.avg_percent_of_workflow},"
                f"{node_perf.total_time_seconds}"
            )

        return "\n".join(lines)


def identify_bottlenecks(workflows: List[Workflow]) -> BottleneckAnalysis:
    """
    Identify performance bottlenecks by analyzing node execution times.

    Analyzes all child nodes across workflows to identify which nodes
    consume the most time and have the highest variability.

    Args:
        workflows: List of Workflow objects to analyze

    Returns:
        BottleneckAnalysis with ranked node performance metrics

    Raises:
        ValueError: If no valid workflows are provided
    """
    if not workflows:
        raise ValueError("No valid workflows to analyze (empty list)")

    # Collect all node executions by node name
    node_executions: dict[str, list[tuple[Trace, float]]] = {}

    for workflow in workflows:
        workflow_duration = workflow.total_duration
        if workflow_duration == 0:
            continue  # Skip workflows with zero duration

        # Iterate through all child nodes (excluding root)
        for node_name, node_traces in workflow.nodes.items():
            if node_name not in node_executions:
                node_executions[node_name] = []

            for trace in node_traces:
                # Calculate percent of workflow time
                percent_of_workflow = (
                    (trace.duration_seconds / workflow_duration) * 100.0
                    if workflow_duration > 0
                    else 0.0
                )
                node_executions[node_name].append((trace, percent_of_workflow))

    # If no nodes found, return empty analysis
    if not node_executions:
        return BottleneckAnalysis(
            node_performances=[], primary_bottleneck=None, top_3_bottlenecks=[]
        )

    # Calculate aggregated statistics for each node
    node_performances = []

    for node_name, executions in node_executions.items():
        durations = np.array([trace.duration_seconds for trace, _ in executions])
        percents = np.array([percent for _, percent in executions])

        node_perf = NodePerformance(
            node_name=node_name,
            execution_count=len(executions),
            avg_duration_seconds=float(np.mean(durations)),
            median_duration_seconds=float(np.median(durations)),
            std_dev_seconds=float(np.std(durations)),
            avg_percent_of_workflow=float(np.mean(percents)),
            total_time_seconds=float(np.sum(durations)),
        )
        node_performances.append(node_perf)

    # Sort by average duration (descending) to identify bottlenecks
    node_performances.sort(key=lambda x: x.avg_duration_seconds, reverse=True)

    # Identify primary bottleneck and top 3
    primary_bottleneck = node_performances[0].node_name if node_performances else None
    top_3_bottlenecks = [n.node_name for n in node_performances[:3]]

    return BottleneckAnalysis(
        node_performances=node_performances,
        primary_bottleneck=primary_bottleneck,
        top_3_bottlenecks=top_3_bottlenecks,
    )


# ============================================================================
# Phase 4: Parallel Execution Verification
# ============================================================================


@dataclass
class ParallelExecutionEvidence:
    """
    Evidence for parallel vs sequential validator execution.

    Attributes:
        parallel_confirmed_count: Number of workflows with parallel execution
        sequential_count: Number of workflows with sequential execution
        avg_start_time_delta_seconds: Average time difference between validator starts
        avg_sequential_time_seconds: Average time if validators ran sequentially
        avg_parallel_time_seconds: Average time with parallel execution
        avg_time_savings_seconds: Average time saved by parallel execution
        is_parallel: True if majority of workflows show parallel execution
        confidence: Confidence level ('high', 'medium', 'low', 'none')
    """

    parallel_confirmed_count: int
    sequential_count: int
    avg_start_time_delta_seconds: float
    avg_sequential_time_seconds: float
    avg_parallel_time_seconds: float
    avg_time_savings_seconds: float
    is_parallel: bool
    confidence: str

    def to_csv(self) -> str:
        """
        Export parallel execution evidence to CSV format.

        Returns:
            CSV-formatted string with parallel execution metrics
        """
        lines = ["metric,value,unit,interpretation"]
        lines.append(
            f"parallel_confirmed_count,{self.parallel_confirmed_count},workflows,"
            "workflows with parallel validators"
        )
        lines.append(
            f"sequential_count,{self.sequential_count},workflows,"
            "workflows with sequential validators"
        )
        lines.append(
            f"avg_start_time_delta_seconds,{self.avg_start_time_delta_seconds},seconds,"
            "avg time between first and last validator start"
        )
        lines.append(
            f"avg_sequential_time_seconds,{self.avg_sequential_time_seconds},seconds,"
            "avg time if validators ran sequentially"
        )
        lines.append(
            f"avg_parallel_time_seconds,{self.avg_parallel_time_seconds},seconds,"
            "avg time with parallel execution"
        )
        lines.append(
            f"avg_time_savings_seconds,{self.avg_time_savings_seconds},seconds,"
            "avg time saved by parallelization"
        )
        lines.append(
            f"is_parallel,{self.is_parallel},boolean,parallel execution verdict"
        )
        lines.append(
            f"confidence,{self.confidence},level,confidence in verdict (high/medium/low/none)"
        )
        return "\n".join(lines)


# Validator node names to detect
VALIDATOR_NAMES = {"meta_evaluation", "normative_validation", "simulated_testing"}

# Threshold for parallel detection (seconds)
PARALLEL_START_THRESHOLD = 5.0  # Validators starting within 5s = parallel


def verify_parallel_execution(workflows: List[Workflow]) -> ParallelExecutionEvidence:
    """
    Verify whether validator nodes execute in parallel across workflows.

    Analyzes validator start times to determine if the three validators
    (meta_evaluation, normative_validation, simulated_testing) run in
    parallel or sequential mode, and calculates time savings.

    Args:
        workflows: List of Workflow objects to analyze

    Returns:
        ParallelExecutionEvidence with parallel execution metrics

    Raises:
        ValueError: If no valid workflows are provided
    """
    if not workflows:
        raise ValueError("No valid workflows to analyze (empty list)")

    parallel_count = 0
    sequential_count = 0
    all_start_deltas: List[float] = []
    all_sequential_times: List[float] = []
    all_parallel_times: List[float] = []
    all_time_savings: List[float] = []

    for workflow in workflows:
        # Find validator nodes
        validator_traces: List[Trace] = []
        for node_name, traces in workflow.nodes.items():
            if node_name in VALIDATOR_NAMES:
                validator_traces.extend(traces)

        # Skip workflows without validators
        if len(validator_traces) < 2:
            continue

        # Check if all validators have start times
        if not all(t.start_time is not None for t in validator_traces):
            continue

        # Sort validators by start time
        sorted_validators = sorted(validator_traces, key=lambda t: t.start_time)  # type: ignore

        # Calculate start time deltas
        start_times = [t.start_time for t in sorted_validators]
        first_start = start_times[0]
        last_start = start_times[-1]

        # Calculate max delta between first and last validator start
        start_delta = (last_start - first_start).total_seconds()  # type: ignore
        all_start_deltas.append(start_delta)

        # Calculate sequential vs parallel time
        durations = [t.duration_seconds for t in validator_traces]
        sequential_time = sum(durations)  # Sum if sequential
        parallel_time = max(durations)  # Max if parallel
        time_savings = sequential_time - parallel_time

        all_sequential_times.append(sequential_time)
        all_parallel_times.append(parallel_time)
        all_time_savings.append(time_savings)

        # Determine if this workflow shows parallel execution
        if start_delta <= PARALLEL_START_THRESHOLD:
            parallel_count += 1
        else:
            sequential_count += 1

    # Handle case where no workflows had validators
    if parallel_count == 0 and sequential_count == 0:
        return ParallelExecutionEvidence(
            parallel_confirmed_count=0,
            sequential_count=0,
            avg_start_time_delta_seconds=0.0,
            avg_sequential_time_seconds=0.0,
            avg_parallel_time_seconds=0.0,
            avg_time_savings_seconds=0.0,
            is_parallel=False,
            confidence="none",
        )

    # Calculate averages
    avg_start_delta = float(np.mean(all_start_deltas)) if all_start_deltas else 0.0
    avg_sequential_time = (
        float(np.mean(all_sequential_times)) if all_sequential_times else 0.0
    )
    avg_parallel_time = (
        float(np.mean(all_parallel_times)) if all_parallel_times else 0.0
    )
    avg_time_savings = float(np.mean(all_time_savings)) if all_time_savings else 0.0

    # Determine overall parallel verdict
    total_workflows = parallel_count + sequential_count
    is_parallel = parallel_count > sequential_count

    # Determine confidence level
    if total_workflows == 0:
        confidence = "none"
    elif parallel_count == total_workflows or sequential_count == total_workflows:
        confidence = "high"  # All workflows show same pattern
    elif max(parallel_count, sequential_count) / total_workflows >= 0.8:
        confidence = "medium"  # 80%+ consistency
    else:
        confidence = "low"  # Mixed results

    return ParallelExecutionEvidence(
        parallel_confirmed_count=parallel_count,
        sequential_count=sequential_count,
        avg_start_time_delta_seconds=avg_start_delta,
        avg_sequential_time_seconds=avg_sequential_time,
        avg_parallel_time_seconds=avg_parallel_time,
        avg_time_savings_seconds=avg_time_savings,
        is_parallel=is_parallel,
        confidence=confidence,
    )
