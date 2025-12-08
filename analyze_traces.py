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
