"""
LangSmith Trace Failure Pattern Analysis Tool - Phase 3C

This module provides failure pattern analysis capabilities for LangSmith trace exports.
Detects failures, analyzes retry sequences, and assesses quality risks.

Following PDCA (Plan-Do-Check-Act) methodology with TDD approach.

Author: Generated with Claude Code (PDCA Framework)
Date: 2025-12-09
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import re
from analyze_traces import Trace, Workflow


# ============================================================================
# Configuration Constants
# ============================================================================

# Status values indicating failure
FAILURE_STATUSES = {"error", "failed", "cancelled"}
SUCCESS_STATUSES = {"success"}

# Retry detection heuristics
RETRY_DETECTION_CONFIG = {
    "max_time_window_seconds": 300,  # 5 min window for retry detection
    "same_node_threshold": 2,  # 2+ executions = potential retry
}

# Error classification patterns (regex)
ERROR_PATTERNS = {
    "validation_failure": r"validation.*fail|invalid.*spec",
    "api_timeout": r"timeout|timed out",
    "import_error": r"import.*fail|import.*error",
    "llm_error": r"model.*error|generation.*fail|token.*limit",
    "unknown": r".*",  # Catch-all
}


# ============================================================================
# Core Data Structures
# ============================================================================


@dataclass
class FailureInstance:
    """Single failure occurrence."""

    trace_id: str
    trace_name: str
    workflow_id: str
    error_message: Optional[str]
    error_type: str  # Classified from ERROR_PATTERNS
    timestamp: Optional[datetime]


@dataclass
class RetrySequence:
    """Detected retry sequence."""

    node_name: str
    workflow_id: str
    attempt_count: int
    attempts: List[Trace]  # Ordered by start_time
    final_status: str  # 'success' or 'failed'
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
    caught_issues_count: int  # Failures detected
    pass_rate_percent: float
    is_necessary: bool  # Based on redundancy analysis


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


# ============================================================================
# Failure Detection Functions
# ============================================================================


def detect_failures(workflow: Workflow) -> List[FailureInstance]:
    """
    Detect all failures in workflow using trace.status and trace.error.

    Args:
        workflow: Workflow to analyze

    Returns:
        List of FailureInstance objects
    """
    failures = []

    for trace in workflow.all_traces:
        if trace.status in FAILURE_STATUSES:
            error_type = classify_error(trace.error)
            failure = FailureInstance(
                trace_id=trace.id,
                trace_name=trace.name,
                workflow_id=workflow.root_trace.id,
                error_message=trace.error,
                error_type=error_type,
                timestamp=trace.start_time,
            )
            failures.append(failure)

    return failures


def classify_error(error_message: Optional[str]) -> str:
    """
    Classify error into type using regex patterns.

    Args:
        error_message: Error message to classify

    Returns:
        Error type string
    """
    if not error_message:
        return "unknown"

    error_lower = error_message.lower()

    # Try each pattern (order matters - more specific first)
    for error_type, pattern in ERROR_PATTERNS.items():
        if error_type == "unknown":
            continue  # Skip catch-all for now
        if re.search(pattern, error_lower):
            return error_type

    return "unknown"


# ============================================================================
# Retry Detection Functions
# ============================================================================


def detect_retry_sequences(workflow: Workflow) -> List[RetrySequence]:
    """
    Detect retry sequences using heuristics:
    - Multiple executions of same node within time window
    - Ordered by start_time

    Args:
        workflow: Workflow to analyze

    Returns:
        List of RetrySequence objects
    """
    # Group traces by node name
    node_traces: Dict[str, List[Trace]] = {}
    for trace in workflow.all_traces:
        if trace.name not in node_traces:
            node_traces[trace.name] = []
        node_traces[trace.name].append(trace)

    retry_sequences = []

    for node_name, traces in node_traces.items():
        if len(traces) < RETRY_DETECTION_CONFIG["same_node_threshold"]:
            continue

        # Sort by start_time
        sorted_traces = sorted(traces, key=lambda t: t.start_time)

        # Check if traces are within time window
        first_start = sorted_traces[0].start_time
        last_start = sorted_traces[-1].start_time
        time_diff = (last_start - first_start).total_seconds()

        if time_diff <= RETRY_DETECTION_CONFIG["max_time_window_seconds"]:
            # This looks like a retry sequence
            final_status = sorted_traces[-1].status
            total_duration = sum(t.duration_seconds for t in sorted_traces)

            retry_seq = RetrySequence(
                node_name=node_name,
                workflow_id=workflow.root_trace.id,
                attempt_count=len(sorted_traces),
                attempts=sorted_traces,
                final_status=final_status,
                total_duration_seconds=total_duration,
            )
            retry_sequences.append(retry_seq)

    return retry_sequences


def calculate_retry_success_rate(
    retry_sequences: List[RetrySequence],
) -> Optional[float]:
    """
    Calculate % of retries that eventually succeed.

    Args:
        retry_sequences: List of RetrySequence objects

    Returns:
        Success rate as percentage, or None if no retries
    """
    if not retry_sequences:
        return None

    successful_retries = sum(
        1 for seq in retry_sequences if seq.final_status in SUCCESS_STATUSES
    )

    return (successful_retries / len(retry_sequences)) * 100.0


# ============================================================================
# Node Failure Analysis Functions
# ============================================================================


def analyze_node_failures(workflows: List[Workflow]) -> List[NodeFailureStats]:
    """
    Analyze failure patterns by node type across workflows.

    Args:
        workflows: List of Workflow objects

    Returns:
        List of NodeFailureStats sorted by failure_rate descending
    """
    # Aggregate by node name
    node_data: Dict[str, Dict[str, Any]] = {}

    for workflow in workflows:
        # Detect all retry sequences for this workflow
        retry_sequences = detect_retry_sequences(workflow)

        for trace in workflow.all_traces:
            node_name = trace.name
            if node_name not in node_data:
                node_data[node_name] = {
                    "total_executions": 0,
                    "failure_count": 0,
                    "success_count": 0,
                    "retry_sequences": 0,
                    "error_types": {},
                }

            node_data[node_name]["total_executions"] += 1

            if trace.status in FAILURE_STATUSES:
                node_data[node_name]["failure_count"] += 1
                # Track error type
                error_type = classify_error(trace.error)
                if error_type not in node_data[node_name]["error_types"]:
                    node_data[node_name]["error_types"][error_type] = 0
                node_data[node_name]["error_types"][error_type] += 1
            elif trace.status in SUCCESS_STATUSES:
                node_data[node_name]["success_count"] += 1

        # Count retry sequences per node
        for retry_seq in retry_sequences:
            if retry_seq.node_name in node_data:
                node_data[retry_seq.node_name]["retry_sequences"] += 1

    # Create NodeFailureStats objects
    stats_list = []
    for node_name, data in node_data.items():
        total_exec = data["total_executions"]
        failure_count = data["failure_count"]
        failure_rate = (failure_count / total_exec * 100.0) if total_exec > 0 else 0.0

        # Calculate avg retries when failing
        retry_sequences = data["retry_sequences"]
        avg_retries = (retry_sequences / failure_count) if failure_count > 0 else 0.0

        stats = NodeFailureStats(
            node_name=node_name,
            total_executions=total_exec,
            failure_count=failure_count,
            success_count=data["success_count"],
            failure_rate_percent=failure_rate,
            retry_sequences_detected=retry_sequences,
            avg_retries_when_failing=avg_retries,
            common_error_types=data["error_types"],
        )
        stats_list.append(stats)

    # Sort by failure rate descending
    stats_list.sort(key=lambda s: s.failure_rate_percent, reverse=True)

    return stats_list


# ============================================================================
# Main Analysis Function
# ============================================================================


def analyze_failures(workflows: List[Workflow]) -> FailureAnalysisResults:
    """
    Perform complete failure pattern analysis.
    Main entry point for Phase 3C.

    Args:
        workflows: List of Workflow objects to analyze

    Returns:
        FailureAnalysisResults with complete analysis
    """
    if not workflows:
        return FailureAnalysisResults(
            total_workflows=0,
            successful_workflows=0,
            failed_workflows=0,
            overall_success_rate_percent=0.0,
            node_failure_stats=[],
            highest_failure_node=None,
            error_type_distribution={},
            most_common_error_type=None,
            total_retry_sequences=0,
            retry_sequences=[],
            retry_success_rate_percent=None,
            avg_cost_of_retries=None,
            validator_analyses=[],
            redundant_validators=[],
            quality_risks_at_scale=[],
        )

    # Detect all failures
    all_failures = []
    for workflow in workflows:
        failures = detect_failures(workflow)
        all_failures.extend(failures)

    # Detect all retry sequences
    all_retry_sequences = []
    for workflow in workflows:
        retries = detect_retry_sequences(workflow)
        all_retry_sequences.extend(retries)

    # Calculate overall success rate
    total_workflows = len(workflows)
    failed_workflows = sum(
        1 for workflow in workflows if workflow.root_trace.status in FAILURE_STATUSES
    )
    successful_workflows = total_workflows - failed_workflows
    overall_success_rate = (
        (successful_workflows / total_workflows * 100.0) if total_workflows > 0 else 0.0
    )

    # Analyze node failures
    node_failure_stats = analyze_node_failures(workflows)
    highest_failure_node = (
        node_failure_stats[0].node_name if node_failure_stats else None
    )

    # Aggregate error types
    error_type_distribution: Dict[str, int] = {}
    for failure in all_failures:
        error_type = failure.error_type
        if error_type not in error_type_distribution:
            error_type_distribution[error_type] = 0
        error_type_distribution[error_type] += 1

    most_common_error = (
        max(error_type_distribution, key=error_type_distribution.get)
        if error_type_distribution
        else None
    )

    # Calculate retry success rate
    retry_success_rate = calculate_retry_success_rate(all_retry_sequences)

    # Placeholder for validator analysis (not yet implemented)
    validator_analyses: List[ValidatorEffectivenessAnalysis] = []
    redundant_validators: List[str] = []

    # Placeholder for quality risks (not yet implemented)
    quality_risks_at_scale: List[str] = []

    return FailureAnalysisResults(
        total_workflows=total_workflows,
        successful_workflows=successful_workflows,
        failed_workflows=failed_workflows,
        overall_success_rate_percent=overall_success_rate,
        node_failure_stats=node_failure_stats,
        highest_failure_node=highest_failure_node,
        error_type_distribution=error_type_distribution,
        most_common_error_type=most_common_error,
        total_retry_sequences=len(all_retry_sequences),
        retry_sequences=all_retry_sequences,
        retry_success_rate_percent=retry_success_rate,
        avg_cost_of_retries=None,  # Not yet implemented
        validator_analyses=validator_analyses,
        redundant_validators=redundant_validators,
        quality_risks_at_scale=quality_risks_at_scale,
    )
