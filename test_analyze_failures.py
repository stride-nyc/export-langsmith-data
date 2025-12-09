"""
Test suite for Phase 3C: Failure Pattern Analysis

Following TDD methodology - tests written FIRST, then implementation.
Tests for failure detection, retry analysis, and quality risk assessment.

Author: Generated with Claude Code (PDCA Framework)
Date: 2025-12-09
"""

import pytest
from datetime import datetime, timezone, timedelta
from analyze_failures import (
    detect_failures,
    classify_error,
    detect_retry_sequences,
    calculate_retry_success_rate,
    FailureInstance,
    RetrySequence,
)
from analyze_traces import Trace, Workflow


class TestFailureDetection:
    """Test failure detection from traces."""

    def test_detect_failures_single_failure(self):
        """Test detecting a single failure in workflow."""
        root = Trace(
            id="root-1",
            name="LangGraph",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            duration_seconds=300.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=["child-1"],
            inputs={},
            outputs={},
            error=None,
        )

        child = Trace(
            id="child-1",
            name="Validator",
            start_time=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 2, 0, tzinfo=timezone.utc),
            duration_seconds=60.0,
            status="error",
            run_type="chain",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error="Validation failed: invalid spec",
        )

        workflow = Workflow(
            root_trace=root,
            nodes={"Validator": [child]},
            all_traces=[root, child],
        )

        result = detect_failures(workflow)

        assert result is not None
        assert len(result) == 1
        assert result[0].trace_id == "child-1"
        assert result[0].trace_name == "Validator"
        assert result[0].error_type == "validation_failure"

    def test_detect_failures_no_failures(self):
        """Test workflow with no failures."""
        root = Trace(
            id="root-2",
            name="LangGraph",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            duration_seconds=300.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        workflow = Workflow(
            root_trace=root,
            nodes={},
            all_traces=[root],
        )

        result = detect_failures(workflow)

        assert result is not None
        assert len(result) == 0

    def test_classify_error_validation_failure(self):
        """Test classifying validation error."""
        error_msg = "Validation failed: invalid specification"
        result = classify_error(error_msg)
        assert result == "validation_failure"

    def test_classify_error_api_timeout(self):
        """Test classifying timeout error."""
        error_msg = "Request timed out after 30 seconds"
        result = classify_error(error_msg)
        assert result == "api_timeout"

    def test_classify_error_import_error(self):
        """Test classifying import error."""
        error_msg = "Import failed: module not found"
        result = classify_error(error_msg)
        assert result == "import_error"

    def test_classify_error_llm_error(self):
        """Test classifying LLM error."""
        error_msg = "Model generation failed: token limit exceeded"
        result = classify_error(error_msg)
        assert result == "llm_error"

    def test_classify_error_unknown(self):
        """Test classifying unknown error."""
        error_msg = "Something went wrong"
        result = classify_error(error_msg)
        assert result == "unknown"

    def test_classify_error_none(self):
        """Test classifying None error."""
        result = classify_error(None)
        assert result == "unknown"


class TestRetryDetection:
    """Test retry sequence detection."""

    def test_detect_retry_sequences_two_attempts(self):
        """Test detecting retry sequence with 2 attempts."""
        root = Trace(
            id="root-1",
            name="LangGraph",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            duration_seconds=300.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=["attempt-1", "attempt-2"],
            inputs={},
            outputs={},
            error=None,
        )

        attempt1 = Trace(
            id="attempt-1",
            name="ChatModel",
            start_time=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 1, 30, tzinfo=timezone.utc),
            duration_seconds=30.0,
            status="error",
            run_type="llm",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error="Timeout",
        )

        attempt2 = Trace(
            id="attempt-2",
            name="ChatModel",
            start_time=datetime(2025, 1, 1, 12, 1, 35, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 2, 5, tzinfo=timezone.utc),
            duration_seconds=30.0,
            status="success",
            run_type="llm",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        workflow = Workflow(
            root_trace=root,
            nodes={"ChatModel": [attempt1, attempt2]},
            all_traces=[root, attempt1, attempt2],
        )

        result = detect_retry_sequences(workflow)

        assert result is not None
        assert len(result) == 1
        assert result[0].node_name == "ChatModel"
        assert result[0].attempt_count == 2
        assert result[0].final_status == "success"
        assert result[0].total_duration_seconds == 60.0

    def test_detect_retry_sequences_no_retries(self):
        """Test workflow with no retry sequences."""
        root = Trace(
            id="root-2",
            name="LangGraph",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            duration_seconds=300.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=["child-1"],
            inputs={},
            outputs={},
            error=None,
        )

        child = Trace(
            id="child-1",
            name="ChatModel",
            start_time=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 2, 0, tzinfo=timezone.utc),
            duration_seconds=60.0,
            status="success",
            run_type="llm",
            parent_id="root-2",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        workflow = Workflow(
            root_trace=root,
            nodes={"ChatModel": [child]},
            all_traces=[root, child],
        )

        result = detect_retry_sequences(workflow)

        assert result is not None
        assert len(result) == 0

    def test_calculate_retry_success_rate_all_succeed(self):
        """Test retry success rate when all retries succeed."""
        retry1 = RetrySequence(
            node_name="ChatModel",
            workflow_id="wf-1",
            attempt_count=2,
            attempts=[],
            final_status="success",
            total_duration_seconds=60.0,
        )

        retry2 = RetrySequence(
            node_name="Validator",
            workflow_id="wf-2",
            attempt_count=3,
            attempts=[],
            final_status="success",
            total_duration_seconds=90.0,
        )

        result = calculate_retry_success_rate([retry1, retry2])

        assert result is not None
        assert result == pytest.approx(100.0, abs=0.01)

    def test_calculate_retry_success_rate_partial_success(self):
        """Test retry success rate with partial success."""
        retry1 = RetrySequence(
            node_name="ChatModel",
            workflow_id="wf-1",
            attempt_count=2,
            attempts=[],
            final_status="success",
            total_duration_seconds=60.0,
        )

        retry2 = RetrySequence(
            node_name="Validator",
            workflow_id="wf-2",
            attempt_count=3,
            attempts=[],
            final_status="error",
            total_duration_seconds=90.0,
        )

        result = calculate_retry_success_rate([retry1, retry2])

        assert result is not None
        assert result == pytest.approx(50.0, abs=0.01)

    def test_calculate_retry_success_rate_empty_list(self):
        """Test retry success rate with empty list."""
        result = calculate_retry_success_rate([])
        assert result is None


# Run tests with: pytest test_analyze_failures.py -v
