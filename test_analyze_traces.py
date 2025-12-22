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
            name="process_data",
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
            root_trace=root, nodes={"process_data": [child]}, all_traces=[root, child]
        )

        # Assert
        assert workflow.root_trace.id == "root-1"
        assert "process_data" in workflow.nodes
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
                            "name": "process_data",
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
            min_threshold=7.0,
            max_threshold=23.0,
            outliers_above_max=["trace-1", "trace-2"],
            outliers_below_min=[],
            percent_within_range=85.0,
        )

        # Assert
        assert dist.p50_minutes == 15.0
        assert dist.p95_minutes == 22.0
        assert len(dist.outliers_above_max) == 2
        assert dist.percent_within_range == 85.0

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

        # Act - Test with explicit thresholds (backward compatibility)
        result = analyze_latency_distribution(
            workflows, min_threshold=7.0, max_threshold=23.0
        )

        # Assert
        assert result.p50_minutes == 15.0  # Median of [5, 10, 15, 20, 25]
        assert result.min_minutes == 5.0
        assert result.max_minutes == 25.0
        assert len(result.outliers_below_min) == 1  # 5 min trace
        assert len(result.outliers_above_max) == 1  # 25 min trace

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


class TestBottleneckIdentification:
    """Test bottleneck identification analysis (Phase 3)."""

    def test_node_performance_dataclass_creation(self):
        """Test creating a NodePerformance with all fields."""
        from analyze_traces import NodePerformance

        # Arrange & Act
        node_perf = NodePerformance(
            node_name="process_data",
            execution_count=100,
            avg_duration_seconds=180.5,
            median_duration_seconds=175.2,
            std_dev_seconds=45.3,
            avg_percent_of_workflow=15.2,
            total_time_seconds=18050.0,
        )

        # Assert
        assert node_perf.node_name == "process_data"
        assert node_perf.execution_count == 100
        assert node_perf.avg_duration_seconds == 180.5
        assert node_perf.avg_percent_of_workflow == 15.2

    def test_bottleneck_analysis_dataclass_creation(self):
        """Test creating a BottleneckAnalysis with all fields."""
        from analyze_traces import NodePerformance, BottleneckAnalysis

        # Arrange
        node1 = NodePerformance(
            node_name="transform_output",
            execution_count=100,
            avg_duration_seconds=250.8,
            median_duration_seconds=245.0,
            std_dev_seconds=60.2,
            avg_percent_of_workflow=21.1,
            total_time_seconds=25080.0,
        )

        node2 = NodePerformance(
            node_name="process_data",
            execution_count=100,
            avg_duration_seconds=180.5,
            median_duration_seconds=175.2,
            std_dev_seconds=45.3,
            avg_percent_of_workflow=15.2,
            total_time_seconds=18050.0,
        )

        # Act
        analysis = BottleneckAnalysis(
            node_performances=[node1, node2],
            primary_bottleneck="transform_output",
            top_3_bottlenecks=["transform_output", "process_data"],
        )

        # Assert
        assert len(analysis.node_performances) == 2
        assert analysis.primary_bottleneck == "transform_output"
        assert analysis.top_3_bottlenecks[0] == "transform_output"

    def test_identify_bottlenecks_basic(self):
        """Test basic bottleneck identification with simple workflow."""
        from analyze_traces import Trace, Workflow, identify_bottlenecks

        # Arrange - Create workflow with multiple nodes
        root = Trace(
            id="root-1",
            name="LangGraph",
            start_time=None,
            end_time=None,
            duration_seconds=600.0,  # 10 min total
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=["node-1", "node-2", "node-3"],
            inputs={},
            outputs={},
            error=None,
        )

        node1 = Trace(
            id="node-1",
            name="process_data",
            start_time=None,
            end_time=None,
            duration_seconds=200.0,  # 33% of workflow
            status="success",
            run_type="chain",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        node2 = Trace(
            id="node-2",
            name="transform_output",
            start_time=None,
            end_time=None,
            duration_seconds=300.0,  # 50% of workflow (bottleneck)
            status="success",
            run_type="chain",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        node3 = Trace(
            id="node-3",
            name="import_step",
            start_time=None,
            end_time=None,
            duration_seconds=100.0,  # 17% of workflow
            status="success",
            run_type="chain",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        workflow = Workflow(
            root_trace=root,
            nodes={
                "process_data": [node1],
                "transform_output": [node2],
                "import_step": [node3],
            },
            all_traces=[root, node1, node2, node3],
        )

        # Act
        result = identify_bottlenecks([workflow])

        # Assert
        assert len(result.node_performances) == 3
        assert result.primary_bottleneck == "transform_output"
        assert result.node_performances[0].node_name == "transform_output"
        assert result.node_performances[0].avg_duration_seconds == 300.0
        assert result.node_performances[0].execution_count == 1

    def test_identify_bottlenecks_multiple_workflows(self):
        """Test bottleneck identification across multiple workflows."""
        from analyze_traces import Trace, Workflow, identify_bottlenecks

        # Arrange - Create 3 workflows with same nodes but different durations
        workflows = []

        for i in range(3):
            root = Trace(
                id=f"root-{i}",
                name="LangGraph",
                start_time=None,
                end_time=None,
                duration_seconds=1000.0,
                status="success",
                run_type="chain",
                parent_id=None,
                child_ids=[f"node1-{i}", f"node2-{i}"],
                inputs={},
                outputs={},
                error=None,
            )

            # Node 1 has consistent performance
            node1 = Trace(
                id=f"node1-{i}",
                name="fast_node",
                start_time=None,
                end_time=None,
                duration_seconds=100.0,  # Always 100 seconds
                status="success",
                run_type="chain",
                parent_id=f"root-{i}",
                child_ids=[],
                inputs={},
                outputs={},
                error=None,
            )

            # Node 2 has variable performance (simulating bottleneck)
            node2 = Trace(
                id=f"node2-{i}",
                name="slow_node",
                start_time=None,
                end_time=None,
                duration_seconds=200.0 + (i * 100.0),  # 200, 300, 400 seconds
                status="success",
                run_type="chain",
                parent_id=f"root-{i}",
                child_ids=[],
                inputs={},
                outputs={},
                error=None,
            )

            workflow = Workflow(
                root_trace=root,
                nodes={"fast_node": [node1], "slow_node": [node2]},
                all_traces=[root, node1, node2],
            )
            workflows.append(workflow)

        # Act
        result = identify_bottlenecks(workflows)

        # Assert
        assert len(result.node_performances) == 2
        assert result.primary_bottleneck == "slow_node"

        # Find slow_node in results
        slow_node = next(
            n for n in result.node_performances if n.node_name == "slow_node"
        )
        assert slow_node.execution_count == 3
        assert slow_node.avg_duration_seconds == 300.0  # (200+300+400)/3
        assert slow_node.std_dev_seconds > 0  # Has variability

    def test_identify_bottlenecks_empty_workflows(self):
        """Test handling of empty workflow list."""
        from analyze_traces import identify_bottlenecks

        # Act & Assert
        with pytest.raises(ValueError, match="No valid workflows"):
            identify_bottlenecks([])

    def test_identify_bottlenecks_workflows_without_children(self):
        """Test handling workflows without child nodes."""
        from analyze_traces import Trace, Workflow, identify_bottlenecks

        # Arrange - Workflow with no child nodes
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
        result = identify_bottlenecks([workflow])

        # Assert - Should return empty analysis
        assert len(result.node_performances) == 0
        assert result.primary_bottleneck is None
        assert result.top_3_bottlenecks == []


class TestParallelExecutionVerification:
    """Test parallel execution verification analysis (Phase 4)."""

    def test_parallel_execution_evidence_dataclass_creation(self):
        """Test creating a ParallelExecutionEvidence with all fields."""
        from analyze_traces import ParallelExecutionEvidence

        # Arrange & Act
        evidence = ParallelExecutionEvidence(
            parallel_confirmed_count=85,
            sequential_count=10,
            avg_start_time_delta_seconds=2.5,
            avg_sequential_time_seconds=600.0,
            avg_parallel_time_seconds=350.0,
            avg_time_savings_seconds=250.0,
            is_parallel=True,
            confidence="high",
        )

        # Assert
        assert evidence.parallel_confirmed_count == 85
        assert evidence.sequential_count == 10
        assert evidence.is_parallel is True
        assert evidence.confidence == "high"
        assert evidence.avg_time_savings_seconds == 250.0

    def test_verify_parallel_execution_detects_parallel(self):
        """Test detection of parallel validator execution."""
        from analyze_traces import Trace, Workflow, verify_parallel_execution

        # Arrange - Workflow with 3 validators starting at nearly same time
        root = Trace(
            id="root-1",
            name="LangGraph",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 10, 0, tzinfo=timezone.utc),
            duration_seconds=600.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=["val-1", "val-2", "val-3"],
            inputs={},
            outputs={},
            error=None,
        )

        # All validators start within 2 seconds of each other (parallel)
        validator1 = Trace(
            id="val-1",
            name="meta_evaluation",
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

        validator2 = Trace(
            id="val-2",
            name="normative_validation",
            start_time=datetime(2025, 1, 1, 12, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 3, 10, tzinfo=timezone.utc),
            duration_seconds=130.0,
            status="success",
            run_type="chain",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        validator3 = Trace(
            id="val-3",
            name="simulated_testing",
            start_time=datetime(2025, 1, 1, 12, 1, 2, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 3, 20, tzinfo=timezone.utc),
            duration_seconds=140.0,
            status="success",
            run_type="chain",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        workflow = Workflow(
            root_trace=root,
            nodes={
                "meta_evaluation": [validator1],
                "normative_validation": [validator2],
                "simulated_testing": [validator3],
            },
            all_traces=[root, validator1, validator2, validator3],
        )

        # Act
        result = verify_parallel_execution([workflow])

        # Assert
        assert result.is_parallel is True
        assert result.parallel_confirmed_count == 1
        assert result.sequential_count == 0
        assert result.avg_start_time_delta_seconds < 5.0  # Started within 5 seconds

    def test_verify_parallel_execution_detects_sequential(self):
        """Test detection of sequential validator execution."""
        from analyze_traces import Trace, Workflow, verify_parallel_execution

        # Arrange - Workflow with validators running sequentially
        root = Trace(
            id="root-1",
            name="LangGraph",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 10, 0, tzinfo=timezone.utc),
            duration_seconds=600.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=["val-1", "val-2", "val-3"],
            inputs={},
            outputs={},
            error=None,
        )

        # Validators start sequentially (>5 seconds apart)
        validator1 = Trace(
            id="val-1",
            name="meta_evaluation",
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

        validator2 = Trace(
            id="val-2",
            name="normative_validation",
            start_time=datetime(2025, 1, 1, 12, 3, 10, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 5, 20, tzinfo=timezone.utc),
            duration_seconds=130.0,
            status="success",
            run_type="chain",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        validator3 = Trace(
            id="val-3",
            name="simulated_testing",
            start_time=datetime(2025, 1, 1, 12, 5, 30, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 7, 50, tzinfo=timezone.utc),
            duration_seconds=140.0,
            status="success",
            run_type="chain",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        workflow = Workflow(
            root_trace=root,
            nodes={
                "meta_evaluation": [validator1],
                "normative_validation": [validator2],
                "simulated_testing": [validator3],
            },
            all_traces=[root, validator1, validator2, validator3],
        )

        # Act
        result = verify_parallel_execution([workflow])

        # Assert
        assert result.is_parallel is False
        assert result.parallel_confirmed_count == 0
        assert result.sequential_count == 1

    def test_verify_parallel_execution_calculates_time_savings(self):
        """Test calculation of time savings from parallel execution."""
        from analyze_traces import Trace, Workflow, verify_parallel_execution

        # Arrange - Parallel execution with known durations
        root = Trace(
            id="root-1",
            name="LangGraph",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 10, 0, tzinfo=timezone.utc),
            duration_seconds=600.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=["val-1", "val-2", "val-3"],
            inputs={},
            outputs={},
            error=None,
        )

        validator1 = Trace(
            id="val-1",
            name="meta_evaluation",
            start_time=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 3, 0, tzinfo=timezone.utc),
            duration_seconds=100.0,  # 100s
            status="success",
            run_type="chain",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        validator2 = Trace(
            id="val-2",
            name="normative_validation",
            start_time=datetime(2025, 1, 1, 12, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 3, 51, tzinfo=timezone.utc),
            duration_seconds=150.0,  # 150s (longest)
            status="success",
            run_type="chain",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        validator3 = Trace(
            id="val-3",
            name="simulated_testing",
            start_time=datetime(2025, 1, 1, 12, 1, 2, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 3, 22, tzinfo=timezone.utc),
            duration_seconds=120.0,  # 120s
            status="success",
            run_type="chain",
            parent_id="root-1",
            child_ids=[],
            inputs={},
            outputs={},
            error=None,
        )

        workflow = Workflow(
            root_trace=root,
            nodes={
                "meta_evaluation": [validator1],
                "normative_validation": [validator2],
                "simulated_testing": [validator3],
            },
            all_traces=[root, validator1, validator2, validator3],
        )

        # Act
        result = verify_parallel_execution([workflow])

        # Assert
        # Sequential time = 100 + 150 + 120 = 370s
        # Parallel time = max(100, 150, 120) = 150s
        # Time savings = 370 - 150 = 220s
        assert result.avg_sequential_time_seconds == 370.0
        assert result.avg_parallel_time_seconds == 150.0
        assert result.avg_time_savings_seconds == 220.0

    def test_verify_parallel_execution_empty_workflows(self):
        """Test handling of empty workflow list."""
        from analyze_traces import verify_parallel_execution

        # Act & Assert
        with pytest.raises(ValueError, match="No valid workflows"):
            verify_parallel_execution([])

    def test_verify_parallel_execution_workflows_without_validators(self):
        """Test handling workflows without validator nodes."""
        from analyze_traces import Trace, Workflow, verify_parallel_execution

        # Arrange - Workflow with no validator nodes
        root = Trace(
            id="root-1",
            name="LangGraph",
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 12, 10, 0, tzinfo=timezone.utc),
            duration_seconds=600.0,
            status="success",
            run_type="chain",
            parent_id=None,
            child_ids=["node-1"],
            inputs={},
            outputs={},
            error=None,
        )

        node1 = Trace(
            id="node-1",
            name="process_data",
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

        workflow = Workflow(
            root_trace=root, nodes={"process_data": [node1]}, all_traces=[root, node1]
        )

        # Act
        result = verify_parallel_execution([workflow])

        # Assert - Should return inconclusive results
        assert result.parallel_confirmed_count == 0
        assert result.sequential_count == 0
        assert result.confidence == "none"


class TestCSVExport:
    """Test CSV export functionality (Phase 5)."""

    def test_latency_distribution_to_csv(self):
        """Test exporting LatencyDistribution to CSV format."""
        from analyze_traces import LatencyDistribution

        # Arrange
        dist = LatencyDistribution(
            p50_minutes=15.0,
            p95_minutes=22.0,
            p99_minutes=25.0,
            min_minutes=8.0,
            max_minutes=27.0,
            mean_minutes=16.5,
            std_dev_minutes=5.2,
            min_threshold=7.0,
            max_threshold=23.0,
            outliers_above_max=["trace-1", "trace-2"],
            outliers_below_min=["trace-3"],
            percent_within_range=85.0,
        )

        # Act
        csv_output = dist.to_csv()

        # Assert
        assert "metric,value_minutes" in csv_output
        assert "p50,15.0" in csv_output
        assert "p95,22.0" in csv_output
        assert "p99,25.0" in csv_output
        assert "mean,16.5" in csv_output
        assert "percent_within_range,85.0" in csv_output
        assert "7.0-23.0 min range" in csv_output  # Verify dynamic threshold text

    def test_bottleneck_analysis_to_csv(self):
        """Test exporting BottleneckAnalysis to CSV format."""
        from analyze_traces import NodePerformance, BottleneckAnalysis

        # Arrange
        node1 = NodePerformance(
            node_name="transform_output",
            execution_count=100,
            avg_duration_seconds=250.8,
            median_duration_seconds=245.0,
            std_dev_seconds=60.2,
            avg_percent_of_workflow=21.1,
            total_time_seconds=25080.0,
        )

        node2 = NodePerformance(
            node_name="process_data",
            execution_count=100,
            avg_duration_seconds=180.5,
            median_duration_seconds=175.2,
            std_dev_seconds=45.3,
            avg_percent_of_workflow=15.2,
            total_time_seconds=18050.0,
        )

        analysis = BottleneckAnalysis(
            node_performances=[node1, node2],
            primary_bottleneck="transform_output",
            top_3_bottlenecks=["transform_output", "process_data"],
        )

        # Act
        csv_output = analysis.to_csv()

        # Assert
        assert (
            "node_name,execution_count,avg_duration_seconds,median_duration_seconds"
            in csv_output
        )
        assert "transform_output,100,250.8,245.0" in csv_output
        assert "process_data,100,180.5,175.2" in csv_output

    def test_parallel_execution_evidence_to_csv(self):
        """Test exporting ParallelExecutionEvidence to CSV format."""
        from analyze_traces import ParallelExecutionEvidence

        # Arrange
        evidence = ParallelExecutionEvidence(
            parallel_confirmed_count=85,
            sequential_count=10,
            avg_start_time_delta_seconds=2.5,
            avg_sequential_time_seconds=600.0,
            avg_parallel_time_seconds=350.0,
            avg_time_savings_seconds=250.0,
            is_parallel=True,
            confidence="high",
        )

        # Act
        csv_output = evidence.to_csv()

        # Assert
        assert "metric,value" in csv_output
        assert "parallel_confirmed_count,85" in csv_output
        assert "sequential_count,10" in csv_output
        assert "avg_time_savings_seconds,250.0" in csv_output
        assert "is_parallel,True" in csv_output
        assert "confidence,high" in csv_output

    def test_csv_export_writes_to_file(self):
        """Test writing CSV output to file."""
        from analyze_traces import LatencyDistribution

        # Arrange
        dist = LatencyDistribution(
            p50_minutes=15.0,
            p95_minutes=22.0,
            p99_minutes=25.0,
            min_minutes=8.0,
            max_minutes=27.0,
            mean_minutes=16.5,
            std_dev_minutes=5.2,
            min_threshold=7.0,
            max_threshold=40.0,
            outliers_above_max=[],
            outliers_below_min=[],
            percent_within_range=90.0,
        )

        # Act
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv", encoding="utf-8"
        ) as f:
            f.write(dist.to_csv())
            temp_path = f.name

        try:
            # Assert - File should exist and contain expected content
            assert os.path.exists(temp_path)
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert "p50,15.0" in content
                assert "p95,22.0" in content
        finally:
            os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
