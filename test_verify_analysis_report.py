"""
Tests for verify_analysis_report.py

This module tests the report verification tool that regenerates
all statistics from trace data for validation purposes.
"""

import pytest
import json
from unittest.mock import patch

from verify_analysis_report import (
    check_value,
    verify_dataset_info,
    verify_latency_distribution,
    verify_bottleneck_analysis,
    verify_parallel_execution,
    generate_summary_report,
    main,
)
from analyze_traces import (
    Trace,
    Workflow,
    TraceDataset,
    LatencyDistribution,
    BottleneckAnalysis,
    NodePerformance,
    ParallelExecutionEvidence,
)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    trace1 = Trace(
        id="trace-1",
        name="LangGraph",
        start_time=None,
        end_time=None,
        duration_seconds=600.0,  # 10 minutes
        status="success",
        run_type="chain",
        parent_id=None,
        child_ids=["child-1", "child-2"],
        inputs={},
        outputs={},
        error=None,
    )

    workflow = Workflow(
        root_trace=trace1,
        nodes={"node1": [trace1]},
        all_traces=[trace1],
    )

    return TraceDataset(
        workflows=[workflow],
        orphan_traces=[],
        metadata={},
        is_hierarchical=True,
    )


class TestCheckValue:
    """Tests for check_value function."""

    def test_check_value_pass(self, capsys):
        """Test check_value with matching values."""
        result = check_value("Test metric", 10.0, 10.05, tolerance=0.1)

        assert result is True
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "Test metric" in captured.out

    def test_check_value_fail(self, capsys):
        """Test check_value with mismatched values."""
        result = check_value("Test metric", 10.0, 15.0, tolerance=0.1)

        assert result is False
        captured = capsys.readouterr()
        assert "FAIL" in captured.out

    def test_check_value_edge_case(self, capsys):
        """Test check_value at tolerance boundary."""
        result = check_value("Test metric", 10.0, 10.1, tolerance=0.1)

        assert result is True  # Exactly at boundary should pass


class TestVerifyDatasetInfo:
    """Tests for verify_dataset_info function."""

    def test_verify_dataset_info_no_expected(self, mock_dataset, capsys):
        """Test dataset info verification without expected values."""
        sample_size = verify_dataset_info(mock_dataset)

        assert sample_size == 1
        captured = capsys.readouterr()
        assert "Sample Size: 1" in captured.out
        assert "Hierarchical Data: True" in captured.out

    def test_verify_dataset_info_with_expected(self, mock_dataset, capsys):
        """Test dataset info verification with expected values."""
        expected = {"sample_size": 1}
        sample_size = verify_dataset_info(mock_dataset, expected)

        assert sample_size == 1
        captured = capsys.readouterr()
        assert "Expected sample size: 1" in captured.out
        assert "Match: True" in captured.out

    def test_verify_dataset_info_mismatch(self, mock_dataset, capsys):
        """Test dataset info verification with mismatched expected values."""
        expected = {"sample_size": 5}
        sample_size = verify_dataset_info(mock_dataset, expected)

        assert sample_size == 1
        captured = capsys.readouterr()
        assert "Match: False" in captured.out


class TestVerifyLatencyDistribution:
    """Tests for verify_latency_distribution function."""

    @patch("verify_analysis_report.analyze_latency_distribution")
    def test_verify_latency_distribution_basic(
        self, mock_analyze, mock_dataset, capsys
    ):
        """Test latency distribution verification without expected values."""
        mock_latency_dist = LatencyDistribution(
            p50_minutes=10.0,
            p95_minutes=20.0,
            p99_minutes=25.0,
            min_minutes=5.0,
            max_minutes=30.0,
            mean_minutes=12.0,
            std_dev_minutes=5.0,
            outliers_above_23min=[],
            outliers_below_7min=[],
            percent_within_7_23_claim=80.0,
        )
        mock_analyze.return_value = mock_latency_dist

        result = verify_latency_distribution(mock_dataset)

        assert result == mock_latency_dist
        captured = capsys.readouterr()
        assert "p50 (median):    10.00 minutes" in captured.out
        assert "Within 7-23 min: 80.0%" in captured.out

    @patch("verify_analysis_report.analyze_latency_distribution")
    def test_verify_latency_distribution_with_expected(
        self, mock_analyze, mock_dataset, capsys
    ):
        """Test latency distribution verification with expected values."""
        mock_latency_dist = LatencyDistribution(
            p50_minutes=10.0,
            p95_minutes=20.0,
            p99_minutes=25.0,
            min_minutes=5.0,
            max_minutes=30.0,
            mean_minutes=12.0,
            std_dev_minutes=5.0,
            outliers_above_23min=[],
            outliers_below_7min=[],
            percent_within_7_23_claim=80.0,
        )
        mock_analyze.return_value = mock_latency_dist

        expected = {
            "latency": {"p50": 10.0, "p95": 20.0, "mean": 12.0},
            "outliers": {
                "below_7_pct": 0.0,
                "within_7_23_pct": 80.0,
                "above_23_pct": 20.0,
            },
        }

        result = verify_latency_distribution(mock_dataset, expected)

        assert result == mock_latency_dist
        captured = capsys.readouterr()
        assert "PASS" in captured.out


class TestVerifyBottleneckAnalysis:
    """Tests for verify_bottleneck_analysis function."""

    @patch("verify_analysis_report.identify_bottlenecks")
    def test_verify_bottleneck_analysis_basic(
        self, mock_identify, mock_dataset, capsys
    ):
        """Test bottleneck analysis verification without expected values."""
        node_perf = NodePerformance(
            node_name="test_node",
            execution_count=5,
            avg_duration_seconds=10.0,
            median_duration_seconds=9.0,
            std_dev_seconds=2.0,
            avg_percent_of_workflow=25.0,
            total_time_seconds=50.0,
        )

        mock_bottleneck = BottleneckAnalysis(
            node_performances=[node_perf],
            primary_bottleneck="test_node",
            top_3_bottlenecks=["test_node"],
        )
        mock_identify.return_value = mock_bottleneck

        result = verify_bottleneck_analysis(mock_dataset)

        assert result == mock_bottleneck
        captured = capsys.readouterr()
        assert "Primary: test_node" in captured.out

    @patch("verify_analysis_report.identify_bottlenecks")
    def test_verify_bottleneck_analysis_with_expected(
        self, mock_identify, mock_dataset, capsys
    ):
        """Test bottleneck analysis verification with expected values."""
        node_perf = NodePerformance(
            node_name="test_node",
            execution_count=5,
            avg_duration_seconds=10.0,
            median_duration_seconds=9.0,
            std_dev_seconds=2.0,
            avg_percent_of_workflow=25.0,
            total_time_seconds=50.0,
        )

        mock_bottleneck = BottleneckAnalysis(
            node_performances=[node_perf],
            primary_bottleneck="test_node",
            top_3_bottlenecks=["test_node"],
        )
        mock_identify.return_value = mock_bottleneck

        expected = {"bottleneck": {"primary": "test_node"}}

        result = verify_bottleneck_analysis(mock_dataset, expected)

        assert result == mock_bottleneck
        captured = capsys.readouterr()
        assert "PASS" in captured.out


class TestVerifyParallelExecution:
    """Tests for verify_parallel_execution function."""

    @patch("verify_analysis_report.analyze_parallel_execution")
    def test_verify_parallel_execution_basic(self, mock_analyze, mock_dataset, capsys):
        """Test parallel execution verification without expected values."""
        mock_parallel = ParallelExecutionEvidence(
            parallel_confirmed_count=3,
            sequential_count=7,
            avg_start_time_delta_seconds=150.0,
            avg_sequential_time_seconds=300.0,
            avg_parallel_time_seconds=100.0,
            avg_time_savings_seconds=200.0,
            is_parallel=False,
            confidence="medium",
        )
        mock_analyze.return_value = mock_parallel

        result = verify_parallel_execution(mock_dataset)

        assert result == mock_parallel
        captured = capsys.readouterr()
        assert "Parallel workflows:   3/1" in captured.out
        assert "Heuristic: Validators starting within 5 seconds" in captured.out

    @patch("verify_analysis_report.analyze_parallel_execution")
    def test_verify_parallel_execution_with_expected(
        self, mock_analyze, mock_dataset, capsys
    ):
        """Test parallel execution verification with expected values."""
        mock_parallel = ParallelExecutionEvidence(
            parallel_confirmed_count=3,
            sequential_count=7,
            avg_start_time_delta_seconds=150.0,
            avg_sequential_time_seconds=300.0,
            avg_parallel_time_seconds=100.0,
            avg_time_savings_seconds=200.0,
            is_parallel=False,
            confidence="medium",
        )
        mock_analyze.return_value = mock_parallel

        expected = {
            "parallel": {
                "parallel_pct": 300.0,  # 3/1 * 100 = 300% (mock has 1 workflow)
                "start_delta_s": 150.0,
                "savings_s": 200.0,
            }
        }

        result = verify_parallel_execution(mock_dataset, expected)

        assert result == mock_parallel
        captured = capsys.readouterr()
        assert "PASS" in captured.out


class TestGenerateSummaryReport:
    """Tests for generate_summary_report function."""

    def test_generate_summary_report(self, mock_dataset, capsys):
        """Test summary report generation."""
        latency_dist = LatencyDistribution(
            p50_minutes=10.0,
            p95_minutes=20.0,
            p99_minutes=25.0,
            min_minutes=5.0,
            max_minutes=30.0,
            mean_minutes=12.0,
            std_dev_minutes=5.0,
            outliers_above_23min=[],
            outliers_below_7min=[],
            percent_within_7_23_claim=80.0,
        )

        node_perf = NodePerformance(
            node_name="test_node",
            execution_count=5,
            avg_duration_seconds=10.0,
            median_duration_seconds=9.0,
            std_dev_seconds=2.0,
            avg_percent_of_workflow=25.0,
            total_time_seconds=50.0,
        )
        bottleneck_analysis = BottleneckAnalysis(
            node_performances=[node_perf],
            primary_bottleneck="test_node",
            top_3_bottlenecks=["test_node", "node2", "node3"],
        )

        parallel_evidence = ParallelExecutionEvidence(
            parallel_confirmed_count=3,
            sequential_count=7,
            avg_start_time_delta_seconds=150.0,
            avg_sequential_time_seconds=300.0,
            avg_parallel_time_seconds=100.0,
            avg_time_savings_seconds=200.0,
            is_parallel=False,
            confidence="medium",
        )

        generate_summary_report(
            mock_dataset, latency_dist, bottleneck_analysis, parallel_evidence
        )

        captured = capsys.readouterr()
        assert "ANALYSIS SUMMARY" in captured.out
        assert "p50 (median): 10.00 min" in captured.out
        assert "Primary: test_node" in captured.out
        assert "Time savings if parallel: 3.3 min" in captured.out


class TestMain:
    """Tests for main function."""

    def test_main_file_not_found(self, capsys):
        """Test main function with non-existent file."""
        with patch("sys.argv", ["verify_analysis_report.py", "nonexistent.json"]):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Error: Input file not found" in captured.out

    @patch("verify_analysis_report.generate_summary_report")
    @patch("verify_analysis_report.verify_parallel_execution")
    @patch("verify_analysis_report.verify_bottleneck_analysis")
    @patch("verify_analysis_report.verify_latency_distribution")
    @patch("verify_analysis_report.verify_dataset_info")
    def test_main_basic(
        self,
        mock_verify_dataset,
        mock_verify_latency,
        mock_verify_bottleneck,
        mock_verify_parallel,
        mock_generate_summary,
        tmp_path,
    ):
        """Test main function with basic usage."""
        # Setup mocks
        mock_verify_dataset.return_value = 1

        mock_latency = LatencyDistribution(
            p50_minutes=10.0,
            p95_minutes=20.0,
            p99_minutes=25.0,
            min_minutes=5.0,
            max_minutes=30.0,
            mean_minutes=12.0,
            std_dev_minutes=5.0,
            outliers_above_23min=[],
            outliers_below_7min=[],
            percent_within_7_23_claim=80.0,
        )
        mock_verify_latency.return_value = mock_latency

        node_perf = NodePerformance(
            node_name="test_node",
            execution_count=5,
            avg_duration_seconds=10.0,
            median_duration_seconds=9.0,
            std_dev_seconds=2.0,
            avg_percent_of_workflow=25.0,
            total_time_seconds=50.0,
        )
        mock_bottleneck = BottleneckAnalysis(
            node_performances=[node_perf],
            primary_bottleneck="test_node",
            top_3_bottlenecks=["test_node", "node2", "node3"],
        )
        mock_verify_bottleneck.return_value = mock_bottleneck

        mock_parallel = ParallelExecutionEvidence(
            parallel_confirmed_count=3,
            sequential_count=7,
            avg_start_time_delta_seconds=150.0,
            avg_sequential_time_seconds=300.0,
            avg_parallel_time_seconds=100.0,
            avg_time_savings_seconds=200.0,
            is_parallel=False,
            confidence="medium",
        )
        mock_verify_parallel.return_value = mock_parallel

        # Create temporary JSON file
        test_file = tmp_path / "test_traces.json"
        test_data = {
            "metadata": {},
            "traces": [
                {
                    "id": "trace-1",
                    "name": "LangGraph",
                    "start_time": "2024-01-01T00:00:00",
                    "end_time": "2024-01-01T00:10:00",
                    "duration_seconds": 600.0,
                    "status": "success",
                    "run_type": "chain",
                    "parent_run_id": None,
                    "child_runs": [],
                    "inputs": {},
                    "outputs": {},
                    "error": None,
                }
            ],
        }
        test_file.write_text(json.dumps(test_data))

        with patch("sys.argv", ["verify_analysis_report.py", str(test_file)]):
            result = main()

        assert result == 0

    @patch("verify_analysis_report.generate_summary_report")
    @patch("verify_analysis_report.verify_parallel_execution")
    @patch("verify_analysis_report.verify_bottleneck_analysis")
    @patch("verify_analysis_report.verify_latency_distribution")
    @patch("verify_analysis_report.verify_dataset_info")
    def test_main_with_expected_values(
        self,
        mock_verify_dataset,
        mock_verify_latency,
        mock_verify_bottleneck,
        mock_verify_parallel,
        mock_generate_summary,
        tmp_path,
    ):
        """Test main function with expected values file."""
        # Setup mocks
        mock_verify_dataset.return_value = 1

        mock_latency = LatencyDistribution(
            p50_minutes=10.0,
            p95_minutes=20.0,
            p99_minutes=25.0,
            min_minutes=5.0,
            max_minutes=30.0,
            mean_minutes=12.0,
            std_dev_minutes=5.0,
            outliers_above_23min=[],
            outliers_below_7min=[],
            percent_within_7_23_claim=80.0,
        )
        mock_verify_latency.return_value = mock_latency

        node_perf = NodePerformance(
            node_name="test_node",
            execution_count=5,
            avg_duration_seconds=10.0,
            median_duration_seconds=9.0,
            std_dev_seconds=2.0,
            avg_percent_of_workflow=25.0,
            total_time_seconds=50.0,
        )
        mock_bottleneck = BottleneckAnalysis(
            node_performances=[node_perf],
            primary_bottleneck="test_node",
            top_3_bottlenecks=["test_node", "node2", "node3"],
        )
        mock_verify_bottleneck.return_value = mock_bottleneck

        mock_parallel = ParallelExecutionEvidence(
            parallel_confirmed_count=3,
            sequential_count=7,
            avg_start_time_delta_seconds=150.0,
            avg_sequential_time_seconds=300.0,
            avg_parallel_time_seconds=100.0,
            avg_time_savings_seconds=200.0,
            is_parallel=False,
            confidence="medium",
        )
        mock_verify_parallel.return_value = mock_parallel

        # Create temporary JSON files
        test_file = tmp_path / "test_traces.json"
        expected_file = tmp_path / "expected.json"

        test_data = {
            "metadata": {},
            "traces": [
                {
                    "id": "trace-1",
                    "name": "LangGraph",
                    "start_time": "2024-01-01T00:00:00",
                    "end_time": "2024-01-01T00:10:00",
                    "duration_seconds": 600.0,
                    "status": "success",
                    "run_type": "chain",
                    "parent_run_id": None,
                    "child_runs": [],
                    "inputs": {},
                    "outputs": {},
                    "error": None,
                }
            ],
        }
        test_file.write_text(json.dumps(test_data))

        expected_data = {"sample_size": 1}
        expected_file.write_text(json.dumps(expected_data))

        with patch(
            "sys.argv",
            [
                "verify_analysis_report.py",
                str(test_file),
                "--expected-values",
                str(expected_file),
            ],
        ):
            result = main()

        assert result == 0
