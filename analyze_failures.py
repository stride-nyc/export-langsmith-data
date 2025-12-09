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
