"""
Unit tests for LangSmith Trace Analysis Tool

Test Strategy: TDD with test fixtures for trace data
- Test data structures (Trace, Workflow, TraceDataset)
- Test data loading from JSON
- Test analysis functions
- Test CSV export

Following PDCA methodology with RED-GREEN-REFACTOR cycles

Author: Generated with Claude Code (PDCA Framework)
Date: 2025-12-08
"""

import pytest
from datetime import datetime, timezone
import json
import tempfile
import os


class TestTraceDataStructure:
    """Test Trace dataclass structure and validation."""

    def test_trace_creation_with_all_fields(self):
        """Test creating a Trace with all fields populated."""
        from analyze_traces import Trace

        # Arrange & Act
        trace = Trace(
            id="trace-123",
            name="test_workflow",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            duration_seconds=300.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=["child-1", "child-2"],
            inputs={"prompt": "test"},
            outputs={"result": "success"},
            error=None,
        )

        # Assert
        assert trace.id == "trace-123"
        assert trace.name == "test_workflow"
        assert trace.duration_seconds == 300.0
        assert len(trace.child_ids) == 2
        assert trace.parent_id is None

    def test_trace_creation_with_minimal_fields(self):
        """Test creating a Trace with minimal required fields."""
        from analyze_traces import Trace

        # Arrange & Act
        trace = Trace(
            id="trace-456",
            name="minimal_workflow",
            start_time=None,
            end_time=None,
            duration_seconds=0.0,
            status="unknown",
            run_type="chain",
            parent_id=None,
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        # Assert
        assert trace.id == "trace-456"
        assert trace.start_time is None
        assert trace.child_ids == []


class TestWorkflowDataStructure:
    """Test Workflow dataclass structure."""

    def test_workflow_creation(self):
        """Test creating a Workflow with root trace and children."""
        from analyze_traces import Trace, Workflow

        # Arrange
        root = Trace(
            id="root-1",
            name="LangGraph",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 10, 0, tzinfo=timezone.utc),
            duration_seconds=600.0,
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
            name="generate_spec",
            start_time=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 3, 0, tzinfo=timezone.utc),
            duration_seconds=120.0,
            status="success",
            run_type="chain",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        # Act
        workflow = Workflow(
            root_trace=root, nodes={"generate_spec": [child]}, all_traces=[root, child]
        )

        # Assert
        assert workflow.root_trace.id == "root-1"
        assert "generate_spec" in workflow.nodes
        assert len(workflow.all_traces) == 2

    def test_workflow_total_duration_property(self):
        """Test that total_duration property returns root trace duration."""
        from analyze_traces import Trace, Workflow

        # Arrange
        root = Trace(
            id="root-1",
            name="LangGraph",
            start_time=None,
            end_time=None,
            duration_seconds=450.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        workflow = Workflow(root_trace=root, nodes={}, all_traces=[root])

        # Act & Assert
        assert workflow.total_duration == 450.0


class TestTraceDatasetStructure:
    """Test TraceDataset dataclass structure."""

    def test_dataset_creation_hierarchical(self):
        """Test creating a TraceDataset with hierarchical workflows."""
        from analyze_traces import Trace, Workflow, TraceDataset

        # Arrange
        root = Trace(
            id="root-1",
            name="LangGraph",
            start_time=None,
            end_time=None,
            duration_seconds=600.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        workflow = Workflow(root_trace=root, nodes={}, all_traces=[root])

        # Act
        dataset = TraceDataset(
            workflows=[workflow],
            orphan_traces=[],
            metadata={"source": "test"},
            is_hierarchical=True,
        )

        # Assert
        assert len(dataset.workflows) == 1
        assert dataset.is_hierarchical is True
        assert dataset.metadata["source"] == "test"

    def test_dataset_creation_flat(self):
        """Test creating a TraceDataset with flat (orphan) traces."""
        from analyze_traces import Trace, TraceDataset

        # Arrange
        orphan = Trace(
            id="orphan-1",
            name="standalone",
            start_time=None,
            end_time=None,
            duration_seconds=100.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        # Act
        dataset = TraceDataset(
            workflows=[],
            orphan_traces=[orphan],
            metadata={},
            is_hierarchical=False,
        )

        # Assert
        assert len(dataset.workflows) == 0
        assert len(dataset.orphan_traces) == 1
        assert dataset.is_hierarchical is False


class TestLoadFromJSON:
    """Test loading trace data from JSON export files."""

    def test_load_from_json_with_hierarchical_data(self):
        """Test loading JSON file with hierarchical child_runs."""
        from analyze_traces import load_from_json

        # Arrange - Create temp JSON file
        test_data = {
            "export_metadata": {
                "export_timestamp": "2025-01-01T12:00:00+00:00",
                "total_traces": 2,
                "langsmith_api_version": "0.4.x",
            },
            "traces": [
                {
                    "id": "root-1",
                    "name": "LangGraph",
                    "start_time": "2025-01-01T12:00:00+00:00",
                    "end_time": "2025-01-01T12:10:00+00:00",
                    "duration_seconds": 600.0,
                    "status": "success",
                    "run_type": "chain",
                    "inputs": {},
                    "outputs": {},
                    "error": None,
                    "child_runs": [
                        {
                            "id": "child-1",
                            "name": "generate_spec",
                            "start_time": "2025-01-01T12:01:00+00:00",
                            "end_time": "2025-01-01T12:03:00+00:00",
                            "duration_seconds": 120.0,
                            "status": "success",
                            "run_type": "chain",
                            "inputs": {},
                            "outputs": {},
                            "error": None,
                            "child_runs": [],
                        }
                    ],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json", encoding="utf-8"
        ) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            # Act
            dataset = load_from_json(temp_path)

            # Assert
            assert dataset.is_hierarchical is True
            assert len(dataset.workflows) == 1
            assert dataset.workflows[0].root_trace.name == "LangGraph"
            assert len(dataset.workflows[0].all_traces) == 2  # root + 1 child
        finally:
            os.remove(temp_path)

    def test_load_from_json_with_flat_data(self):
        """Test loading JSON file with flat traces (no child_runs)."""
        from analyze_traces import load_from_json

        # Arrange
        test_data = {
            "export_metadata": {
                "export_timestamp": "2025-01-01T12:00:00+00:00",
                "total_traces": 1,
                "langsmith_api_version": "0.4.x",
            },
            "traces": [
                {
                    "id": "trace-1",
                    "name": "standalone",
                    "start_time": "2025-01-01T12:00:00+00:00",
                    "end_time": "2025-01-01T12:05:00+00:00",
                    "duration_seconds": 300.0,
                    "status": "success",
                    "run_type": "chain",
                    "inputs": {},
                    "outputs": {},
                    "error": None,
                    "child_runs": [],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json", encoding="utf-8"
        ) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            # Act
            dataset = load_from_json(temp_path)

            # Assert
            assert dataset.is_hierarchical is False
            assert len(dataset.orphan_traces) == 1
            assert dataset.orphan_traces[0].name == "standalone"
        finally:
            os.remove(temp_path)

    def test_load_from_json_file_not_found(self):
        """Test that load_from_json raises error for missing file."""
        from analyze_traces import load_from_json

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            load_from_json("/nonexistent/path/file.json")

    def test_load_from_json_invalid_json(self):
        """Test that load_from_json raises error for invalid JSON."""
        from analyze_traces import load_from_json

        # Arrange - Create file with invalid JSON
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json", encoding="utf-8"
        ) as f:
            f.write("{invalid json")
            temp_path = f.name

        try:
            # Act & Assert
            with pytest.raises(json.JSONDecodeError):
                load_from_json(temp_path)
        finally:
            os.remove(temp_path)


class TestLatencyDistribution:
    """Test latency distribution analysis (Phase 2)."""

    def test_latency_distribution_dataclass_creation(self):
        """Test creating a LatencyDistribution with all fields."""
        from analyze_traces import LatencyDistribution

        # Arrange & Act
        dist = LatencyDistribution(
            p50_minutes=15.0,
            p95_minutes=22.0,
            p99_minutes=25.0,
            min_minutes=8.0,
            max_minutes=27.0,
            mean_minutes=16.5,
            std_dev_minutes=5.2,
            outliers_above_23min=["trace-1", "trace-2"],
            outliers_below_7min=[],
            percent_within_7_23_claim=85.0,
        )

        # Assert
        assert dist.p50_minutes == 15.0
        assert dist.p95_minutes == 22.0
        assert len(dist.outliers_above_23min) == 2
        assert dist.percent_within_7_23_claim == 85.0

    def test_analyze_latency_distribution_basic(self):
        """Test basic latency distribution analysis with valid workflows."""
        from analyze_traces import (
            Trace,
            Workflow,
            analyze_latency_distribution,
        )

        # Arrange - Create test workflows with various durations
        workflows = []
        durations_seconds = [
            600,  # 10 min
            900,  # 15 min
            1200,  # 20 min
            300,  # 5 min (below 7 min)
            1500,  # 25 min (above 23 min)
        ]

        for i, duration in enumerate(durations_seconds):
            root = Trace(
                id=f"root-{i}",
                name="LangGraph",
                start_time=None,
                end_time=None,
                duration_seconds=duration,
                status="success",
                run_type="chain",
                parent_id=None,
                child_ids=[],
                inputs={},
                outputs={},
                error=None,
            )
            workflow = Workflow(root_trace=root, nodes={}, all_traces=[root])
            workflows.append(workflow)

        # Act
        result = analyze_latency_distribution(workflows)

        # Assert
        assert result.p50_minutes == 15.0  # Median of [5, 10, 15, 20, 25]
        assert result.min_minutes == 5.0
        assert result.max_minutes == 25.0
        assert len(result.outliers_below_7min) == 1  # 5 min trace
        assert len(result.outliers_above_23min) == 1  # 25 min trace

    def test_analyze_latency_distribution_filters_zero_duration(self):
        """Test that zero-duration workflows are filtered out."""
        from analyze_traces import (
            Trace,
            Workflow,
            analyze_latency_distribution,
        )

        # Arrange - Mix of valid and zero-duration workflows
        workflows = []

        # Valid workflow
        valid_root = Trace(
            id="valid-1",
            name="LangGraph",
            start_time=None,
            end_time=None,
            duration_seconds=600.0,  # 10 min
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )
        workflows.append(
            Workflow(root_trace=valid_root, nodes={}, all_traces=[valid_root])
        )

        # Zero-duration workflow (should be filtered)
        zero_root = Trace(
            id="zero-1",
            name="LangGraph",
            start_time=None,
            end_time=None,
            duration_seconds=0.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )
        workflows.append(
            Workflow(root_trace=zero_root, nodes={}, all_traces=[zero_root])
        )

        # Act
        result = analyze_latency_distribution(workflows)

        # Assert - Should only analyze the valid workflow
        assert result.min_minutes == 10.0
        assert result.max_minutes == 10.0

    def test_analyze_latency_distribution_calculates_percentiles(self):
        """Test that p50, p95, p99 percentiles are calculated correctly."""
        from analyze_traces import (
            Trace,
            Workflow,
            analyze_latency_distribution,
        )

        # Arrange - Create 100 workflows with durations from 10-20 minutes
        workflows = []
        for i in range(100):
            duration_seconds = 600 + (i * 6)  # 10-19.9 minutes
            root = Trace(
                id=f"root-{i}",
                name="LangGraph",
                start_time=None,
                end_time=None,
                duration_seconds=duration_seconds,
                status="success",
                run_type="chain",
                parent_id=None,
                child_ids=[],
                inputs={},
                outputs={},
                error=None,
            )
            workflow = Workflow(root_trace=root, nodes={}, all_traces=[root])
            workflows.append(workflow)

        # Act
        result = analyze_latency_distribution(workflows)

        # Assert - Check percentiles are reasonable
        assert 10 <= result.p50_minutes <= 20
        assert result.p50_minutes < result.p95_minutes < result.p99_minutes
        assert result.min_minutes <= result.p50_minutes <= result.max_minutes

    def test_analyze_latency_distribution_empty_workflows(self):
        """Test handling of empty workflow list."""
        from analyze_traces import analyze_latency_distribution

        # Act & Assert
        with pytest.raises(ValueError, match="No valid workflows"):
            analyze_latency_distribution([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
