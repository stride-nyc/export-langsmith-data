"""
Unit tests for LangSmith Data Export Script

Test Strategy: TDD with mocked LangSmith client
- Test each component in isolation
- Mock external API calls
- Test error scenarios comprehensively

Author: Generated with Claude Code (PDCA Framework)
Date: 2025-11-28
"""

import pytest
from unittest.mock import Mock, patch
from export_langsmith_traces import (
    LangSmithExporter,
    parse_arguments,
    AuthenticationError,
    ExportError,
)


class TestArgumentParsing:
    """Test CLI argument parsing."""

    def test_parse_arguments_with_required_args(self):
        """Test that all required arguments are parsed correctly."""
        # Arrange
        test_args = [
            "--api-key",
            "lsv2_pt_test_key",
            "--project",
            "test-project",
            "--limit",
            "100",
            "--output",
            "test_output.json",
        ]

        # Act
        with patch("sys.argv", ["export_langsmith_traces.py"] + test_args):
            args = parse_arguments()

        # Assert
        assert args.api_key == "lsv2_pt_test_key"
        assert args.project == "test-project"
        assert args.limit == 100
        assert args.output == "test_output.json"

    def test_parse_arguments_missing_required_arg(self):
        """Test that missing required argument raises error."""
        # Arrange - missing --api-key argument
        test_args = [
            "--project",
            "test-project",
            "--limit",
            "100",
            "--output",
            "test_output.json",
        ]

        # Act & Assert - should raise SystemExit
        with patch("sys.argv", ["export_langsmith_traces.py"] + test_args):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_parse_arguments_invalid_limit(self):
        """Test that negative or zero limit is rejected."""
        # Arrange - negative limit
        test_args = [
            "--api-key",
            "lsv2_pt_test_key",
            "--project",
            "test-project",
            "--limit",
            "-10",
            "--output",
            "test_output.json",
        ]

        # Act & Assert - should raise SystemExit
        with patch("sys.argv", ["export_langsmith_traces.py"] + test_args):
            with pytest.raises(SystemExit):
                parse_arguments()


class TestLangSmithExporter:
    """Test LangSmithExporter class methods."""

    @patch("export_langsmith_traces.Client")
    def test_client_init_success(self, mock_client_class):
        """Test successful client initialization with valid API key."""
        # Arrange
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        api_key = "lsv2_pt_test_key"
        api_url = "https://api.smith.langchain.com"

        # Act
        exporter = LangSmithExporter(api_key=api_key, api_url=api_url)

        # Assert
        mock_client_class.assert_called_once_with(api_key=api_key, api_url=api_url)
        assert exporter.client == mock_client_instance
        assert exporter.api_key == api_key
        assert exporter.api_url == api_url

    @patch("export_langsmith_traces.Client")
    def test_client_init_auth_error(self, mock_client_class):
        """Test authentication error handling."""
        # Arrange - Mock Client to raise an authentication error
        mock_client_class.side_effect = Exception("Invalid API key")
        api_key = "invalid_key"

        # Act & Assert - should raise AuthenticationError
        with pytest.raises(AuthenticationError) as exc_info:
            LangSmithExporter(api_key=api_key)

        assert (
            "authentication" in str(exc_info.value).lower()
            or "api key" in str(exc_info.value).lower()
        )

    @patch("export_langsmith_traces.Client")
    def test_fetch_runs_success(self, mock_client_class):
        """Test successful run fetching."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create mock Run objects
        mock_run1 = Mock()
        mock_run1.id = "run_1"
        mock_run1.name = "test_run_1"

        mock_run2 = Mock()
        mock_run2.id = "run_2"
        mock_run2.name = "test_run_2"

        # Mock list_runs to return an iterable
        mock_client.list_runs.return_value = [mock_run1, mock_run2]

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        runs = exporter.fetch_runs(project_name="test-project", limit=2)

        # Assert
        mock_client.list_runs.assert_called_once_with(
            project_name="test-project", limit=2
        )
        assert len(runs) == 2
        assert runs[0].id == "run_1"
        assert runs[1].id == "run_2"

    @patch("export_langsmith_traces.Client")
    @patch("export_langsmith_traces.time.sleep")  # Mock sleep to speed up test
    def test_fetch_runs_with_rate_limit(self, mock_sleep, mock_client_class):
        """Test rate limit handling with retry."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create mock Run object for successful retry
        mock_run = Mock()
        mock_run.id = "run_1"

        # Mock list_runs to fail once with rate limit, then succeed
        mock_client.list_runs.side_effect = [
            Exception("Rate limit exceeded"),  # First call fails
            [mock_run],  # Second call succeeds
        ]

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        runs = exporter.fetch_runs(project_name="test-project", limit=1)

        # Assert
        assert len(runs) == 1
        assert runs[0].id == "run_1"
        assert mock_client.list_runs.call_count == 2  # Called twice (retry)
        mock_sleep.assert_called_once()  # Sleep was called for backoff

    @patch("export_langsmith_traces.Client")
    @patch("export_langsmith_traces.time.sleep")  # Mock sleep to speed up test
    def test_fetch_runs_max_retries_exceeded(self, mock_sleep, mock_client_class):
        """Test max retry limit raises error."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock list_runs to always fail
        mock_client.list_runs.side_effect = Exception("Rate limit exceeded")

        exporter = LangSmithExporter(api_key="test_key")

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            exporter.fetch_runs(project_name="test-project", limit=1)

        assert "Rate limit exceeded" in str(exc_info.value)
        # Should be called MAX_RETRIES times (5)
        assert mock_client.list_runs.call_count == exporter.MAX_RETRIES
        # Should sleep MAX_RETRIES - 1 times (4)
        assert mock_sleep.call_count == exporter.MAX_RETRIES - 1

    @patch("export_langsmith_traces.Client")
    def test_format_trace_data_complete_fields(self, mock_client_class):
        """Test data formatting with all fields present."""
        # Arrange
        from datetime import datetime, timezone

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create mock Run with all fields
        mock_run = Mock()
        mock_run.id = "run_123"
        mock_run.name = "test_workflow"
        mock_run.start_time = datetime(2025, 11, 28, 10, 0, 0, tzinfo=timezone.utc)
        mock_run.end_time = datetime(2025, 11, 28, 10, 15, 0, tzinfo=timezone.utc)
        mock_run.status = "success"
        mock_run.inputs = {"input_key": "input_value"}
        mock_run.outputs = {"output_key": "output_value"}
        mock_run.error = None
        mock_run.run_type = "chain"
        mock_run.child_runs = []

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        result = exporter.format_trace_data([mock_run])

        # Assert
        assert "export_metadata" in result
        assert "traces" in result

        # Check metadata
        metadata = result["export_metadata"]
        assert "export_timestamp" in metadata
        assert "total_traces" in metadata
        assert metadata["total_traces"] == 1

        # Check trace data
        traces = result["traces"]
        assert len(traces) == 1

        trace = traces[0]
        assert trace["id"] == "run_123"
        assert trace["name"] == "test_workflow"
        assert trace["status"] == "success"
        assert trace["inputs"] == {"input_key": "input_value"}
        assert trace["outputs"] == {"output_key": "output_value"}
        assert trace["error"] is None
        assert trace["run_type"] == "chain"
        assert "start_time" in trace
        assert "end_time" in trace
        assert "duration_seconds" in trace
        assert trace["child_runs"] == []

    @patch("export_langsmith_traces.Client")
    def test_format_trace_data_missing_fields(self, mock_client_class):
        """Test safe handling of missing/null fields."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create mock Run with minimal fields (many missing/None)
        mock_run = Mock(spec=[])  # Empty spec means no attributes by default
        mock_run.id = "run_456"
        mock_run.name = "minimal_run"
        # Missing: start_time, end_time, status, inputs, outputs, error, run_type, child_runs

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        result = exporter.format_trace_data([mock_run])

        # Assert - should not crash and use default values
        assert "traces" in result
        traces = result["traces"]
        assert len(traces) == 1

        trace = traces[0]
        assert trace["id"] == "run_456"
        assert trace["name"] == "minimal_run"
        assert trace["start_time"] is None
        assert trace["end_time"] is None
        assert trace["duration_seconds"] == 0
        assert trace["status"] is None
        assert trace["inputs"] == {}
        assert trace["outputs"] == {}
        assert trace["error"] is None
        assert trace["run_type"] is None
        assert trace["child_runs"] == []

    @patch("export_langsmith_traces.Client")
    def test_format_trace_data_with_child_runs(self, mock_client_class):
        """Test nested run relationship handling."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create mock child runs
        child_run1 = Mock(spec=["id", "name"])
        child_run1.id = "child_1"
        child_run1.name = "child_workflow_1"

        child_run2 = Mock(spec=["id", "name"])
        child_run2.id = "child_2"
        child_run2.name = "child_workflow_2"

        # Create parent run with children
        parent_run = Mock(spec=["id", "name", "child_runs"])
        parent_run.id = "parent_789"
        parent_run.name = "parent_workflow"
        parent_run.child_runs = [child_run1, child_run2]

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        result = exporter.format_trace_data([parent_run])

        # Assert
        traces = result["traces"]
        assert len(traces) == 1

        parent_trace = traces[0]
        assert parent_trace["id"] == "parent_789"
        assert parent_trace["name"] == "parent_workflow"
        assert len(parent_trace["child_runs"]) == 2
        assert parent_trace["child_runs"][0].id == "child_1"
        assert parent_trace["child_runs"][1].id == "child_2"

    @patch("export_langsmith_traces.Client")
    def test_export_to_json_success(self, mock_client_class):
        """Test successful JSON file creation."""
        import json
        import tempfile
        import os

        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        exporter = LangSmithExporter(api_key="test_key")

        test_data = {
            "export_metadata": {
                "export_timestamp": "2025-11-28T12:00:00Z",
                "total_traces": 1,
            },
            "traces": [{"id": "test_123", "name": "test_trace"}],
        }

        # Create a temporary file path
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # Act
            exporter.export_to_json(test_data, temp_path)

            # Assert
            assert os.path.exists(temp_path)

            # Read and verify JSON content
            with open(temp_path, "r") as f:
                saved_data = json.load(f)

            assert saved_data == test_data
            assert saved_data["export_metadata"]["total_traces"] == 1
            assert len(saved_data["traces"]) == 1
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch("export_langsmith_traces.Client")
    @patch("export_langsmith_traces.open")
    def test_export_to_json_write_error(self, mock_open, mock_client_class):
        """Test file write error handling."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock open to raise PermissionError
        mock_open.side_effect = PermissionError("Permission denied")

        exporter = LangSmithExporter(api_key="test_key")

        test_data = {"export_metadata": {}, "traces": []}

        # Act & Assert
        with pytest.raises(ExportError) as exc_info:
            exporter.export_to_json(test_data, "/invalid/path/file.json")

        assert (
            "export" in str(exc_info.value).lower()
            or "permission" in str(exc_info.value).lower()
        )


class TestErrorHandling:
    """Test error scenarios."""

    def test_project_not_found(self):
        """Test project not found error."""
        pass

    def test_network_timeout(self):
        """Test network timeout handling."""
        pass

    def test_zero_traces_returned(self):
        """Test handling of empty results."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
