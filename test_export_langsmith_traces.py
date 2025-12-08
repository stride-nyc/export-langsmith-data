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
    validate_required_args,
    AuthenticationError,
    ExportError,
    RateLimitError,
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
        """Test that missing required argument (no env var fallback) raises error."""
        # Arrange - missing --api-key argument and no env var
        test_args = [
            "--project",
            "test-project",
            "--limit",
            "100",
            "--output",
            "test_output.json",
        ]

        # Act & Assert - parse_arguments with no env vars, then validate should fail
        with patch.dict(
            "os.environ", {}, clear=True
        ):  # Clear all env vars before parsing
            with patch(
                "export_langsmith_traces.load_dotenv"
            ):  # Prevent loading .env file
                with patch("sys.argv", ["export_langsmith_traces.py"] + test_args):
                    args = parse_arguments()
                    # api_key should be None since not provided via CLI or env
                    assert args.api_key is None
                    # Now validate_required_args should raise SystemExit
                    with pytest.raises(SystemExit):
                        validate_required_args(args)

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


class TestValidateRequiredArgs:
    """Test validate_required_args function with various argument combinations."""

    def test_validate_with_all_args_provided(self):
        """Test validation passes when all required args are provided."""
        # Arrange
        from argparse import Namespace

        args = Namespace(
            api_key="test_key", project="test-project", limit=100, output="test.json"
        )

        # Act & Assert - should not raise
        validate_required_args(args)

    def test_validate_missing_api_key(self):
        """Test validation fails when api_key is missing."""
        # Arrange
        from argparse import Namespace

        args = Namespace(
            api_key=None, project="test-project", limit=100, output="test.json"
        )

        # Act & Assert
        with pytest.raises(SystemExit):
            validate_required_args(args)

    def test_validate_missing_project(self):
        """Test validation fails when project is missing."""
        # Arrange
        from argparse import Namespace

        args = Namespace(
            api_key="test_key", project=None, limit=100, output="test.json"
        )

        # Act & Assert
        with pytest.raises(SystemExit):
            validate_required_args(args)

    def test_validate_missing_limit(self):
        """Test validation fails when limit is missing."""
        # Arrange
        from argparse import Namespace

        args = Namespace(
            api_key="test_key", project="test-project", limit=None, output="test.json"
        )

        # Act & Assert
        with pytest.raises(SystemExit):
            validate_required_args(args)

    def test_validate_multiple_missing_args(self):
        """Test validation fails when multiple args are missing."""
        # Arrange
        from argparse import Namespace

        args = Namespace(api_key=None, project=None, limit=None, output="test.json")

        # Act & Assert
        with pytest.raises(SystemExit):
            validate_required_args(args)

    def test_parse_with_env_vars(self):
        """Test that args can be provided via environment variables."""
        # Arrange
        test_args = ["--output", "test.json"]

        # Act
        with patch.dict(
            "os.environ",
            {
                "LANGSMITH_API_KEY": "env_api_key",
                "LANGSMITH_PROJECT": "env_project",
                "LANGSMITH_LIMIT": "200",
            },
        ):
            with patch("sys.argv", ["export_langsmith_traces.py"] + test_args):
                args = parse_arguments()

        # Assert
        assert args.api_key == "env_api_key"
        assert args.project == "env_project"
        assert args.limit == 200
        assert args.output == "test.json"

        # Validate should pass
        validate_required_args(args)

    def test_parse_with_mixed_cli_and_env(self):
        """Test that CLI args override environment variables."""
        # Arrange
        test_args = ["--api-key", "cli_key", "--output", "test.json"]

        # Act
        with patch.dict(
            "os.environ",
            {
                "LANGSMITH_API_KEY": "env_key",
                "LANGSMITH_PROJECT": "env_project",
                "LANGSMITH_LIMIT": "200",
            },
        ):
            with patch("sys.argv", ["export_langsmith_traces.py"] + test_args):
                args = parse_arguments()

        # Assert - CLI should override env
        assert args.api_key == "cli_key"  # From CLI
        assert args.project == "env_project"  # From env
        assert args.limit == 200  # From env

        # Validate should pass
        validate_required_args(args)


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

    @patch("export_langsmith_traces.Client")
    def test_project_not_found(self, mock_client_class):
        """Test project not found error."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock list_runs to raise an error indicating project not found
        mock_client.list_runs.side_effect = Exception("Project 'invalid' not found")

        exporter = LangSmithExporter(api_key="test_key")

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            exporter.fetch_runs(project_name="invalid", limit=10)

        assert (
            "not found" in str(exc_info.value).lower()
            or "project" in str(exc_info.value).lower()
        )

    @patch("export_langsmith_traces.Client")
    @patch("export_langsmith_traces.time.sleep")
    def test_network_timeout(self, mock_sleep, mock_client_class):
        """Test network timeout handling."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock list_runs to always raise timeout error
        mock_client.list_runs.side_effect = TimeoutError("Connection timeout")

        exporter = LangSmithExporter(api_key="test_key")

        # Act & Assert - should retry and eventually raise RateLimitError
        with pytest.raises(RateLimitError) as exc_info:
            exporter.fetch_runs(project_name="test-project", limit=10)

        # Verify the error message contains info about the timeout
        assert "Connection timeout" in str(exc_info.value)

        # Should have retried MAX_RETRIES times
        assert mock_client.list_runs.call_count == exporter.MAX_RETRIES

    @patch("export_langsmith_traces.Client")
    def test_zero_traces_returned(self, mock_client_class):
        """Test handling of empty results."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock list_runs to return empty list
        mock_client.list_runs.return_value = []

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        runs = exporter.fetch_runs(project_name="test-project", limit=10)

        # Assert - should handle gracefully without error
        assert runs == []
        assert len(runs) == 0


class TestPagination:
    """Test pagination logic for handling > 100 record limits."""

    @patch("export_langsmith_traces.Client")
    def test_fetch_runs_single_page(self, mock_client_class):
        """Test fetching runs that fit in a single page (limit <= 100)."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create 50 mock runs
        mock_runs = [Mock(id=f"run_{i}", name=f"test_{i}") for i in range(50)]
        mock_client.list_runs.return_value = iter(mock_runs)

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        runs = exporter.fetch_runs(project_name="test-project", limit=50)

        # Assert
        assert len(runs) == 50
        assert mock_client.list_runs.call_count == 1  # Only one API call
        mock_client.list_runs.assert_called_with(project_name="test-project", limit=50)

    @patch("export_langsmith_traces.Client")
    @patch("export_langsmith_traces.time.sleep")
    def test_fetch_runs_multiple_pages(self, mock_sleep, mock_client_class):
        """Test fetching runs across multiple pages (limit > 100)."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create 250 mock runs
        all_mock_runs = [Mock(id=f"run_{i}", name=f"test_{i}") for i in range(250)]

        # Mock list_runs to return appropriate slices based on limit parameter
        def mock_list_runs(*args, **kwargs):
            limit = kwargs.get("limit", len(all_mock_runs))
            return iter(all_mock_runs[:limit])

        mock_client.list_runs.side_effect = mock_list_runs

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        runs = exporter.fetch_runs(project_name="test-project", limit=250)

        # Assert
        assert len(runs) == 250
        assert mock_client.list_runs.call_count == 3  # 3 pages (100 + 100 + 50)

        # Verify calls were made with correct parameters
        calls = mock_client.list_runs.call_args_list
        # First page requests 100, second requests 200 total (to skip 100), third requests 250 total (to skip 200)
        assert calls[0].kwargs["limit"] == 100
        assert calls[1].kwargs["limit"] == 200
        assert calls[2].kwargs["limit"] == 250

    @patch("export_langsmith_traces.Client")
    @patch("builtins.print")
    def test_fetch_runs_fewer_than_requested(self, mock_print, mock_client_class):
        """Test warning when project has fewer runs than limit."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Only 75 runs available in project
        available_runs = [Mock(id=f"run_{i}", name=f"test_{i}") for i in range(75)]
        mock_client.list_runs.return_value = iter(available_runs)

        exporter = LangSmithExporter(api_key="test_key")

        # Act - Request 150 but only 75 available
        runs = exporter.fetch_runs(project_name="test-project", limit=150)

        # Assert
        assert len(runs) == 75
        # Check that warning was printed
        warning_printed = any(
            "Warning" in str(call) or "warning" in str(call).lower()
            for call in mock_print.call_args_list
        )
        assert warning_printed, "Expected warning about fewer runs than requested"

    @patch("export_langsmith_traces.Client")
    def test_fetch_runs_empty_project(self, mock_client_class):
        """Test handling of project with no runs."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Empty iterator
        mock_client.list_runs.return_value = iter([])

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        runs = exporter.fetch_runs(project_name="test-project", limit=100)

        # Assert
        assert len(runs) == 0
        assert runs == []

    @patch("export_langsmith_traces.Client")
    @patch("export_langsmith_traces.time.sleep")
    def test_fetch_runs_pagination_with_retry(self, mock_sleep, mock_client_class):
        """Test that retry logic works during paginated fetches."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create 150 mock runs
        all_runs = [Mock(id=f"run_{i}", name=f"test_{i}") for i in range(150)]

        # First page succeeds, second page fails once then succeeds
        call_count = [0]

        def mock_list_runs(*args, **kwargs):
            call_count[0] += 1
            limit = kwargs.get("limit", len(all_runs))

            # Second API call (page 2) fails first time
            if call_count[0] == 2:
                raise Exception("Rate limit exceeded")

            return iter(all_runs[:limit])

        mock_client.list_runs.side_effect = mock_list_runs

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        runs = exporter.fetch_runs(project_name="test-project", limit=150)

        # Assert
        assert len(runs) == 150
        # Should be called 3 times: page 1 success, page 2 fail, page 2 retry success
        assert mock_client.list_runs.call_count == 3

    @patch("export_langsmith_traces.Client")
    @patch("export_langsmith_traces.time.sleep")
    def test_fetch_runs_exact_page_boundary(self, mock_sleep, mock_client_class):
        """Test fetching exactly 200 runs (2 full pages)."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create exactly 200 mock runs
        all_mock_runs = [Mock(id=f"run_{i}", name=f"test_{i}") for i in range(200)]

        def mock_list_runs(*args, **kwargs):
            limit = kwargs.get("limit", len(all_mock_runs))
            return iter(all_mock_runs[:limit])

        mock_client.list_runs.side_effect = mock_list_runs

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        runs = exporter.fetch_runs(project_name="test-project", limit=200)

        # Assert
        assert len(runs) == 200
        assert mock_client.list_runs.call_count == 2  # Exactly 2 pages

    @patch("export_langsmith_traces.Client")
    @patch("builtins.print")
    @patch("export_langsmith_traces.time.sleep")
    def test_fetch_runs_shows_progress_multi_page(
        self, mock_sleep, mock_print, mock_client_class
    ):
        """Test that progress messages are shown for multi-page fetches."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        all_runs = [Mock(id=f"run_{i}") for i in range(250)]

        def mock_list_runs(*args, **kwargs):
            limit = kwargs.get("limit", len(all_runs))
            return iter(all_runs[:limit])

        mock_client.list_runs.side_effect = mock_list_runs

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        runs = exporter.fetch_runs(project_name="test-project", limit=250)

        # Assert
        assert len(runs) == 250

        # Check progress messages were printed
        print_calls = [str(call) for call in mock_print.call_args_list]

        # Should show pagination info or page progress
        page_mentions = [
            call for call in print_calls if "page" in call.lower() or "Page" in call
        ]
        assert len(page_mentions) > 0, "Expected pagination progress messages"


class TestIntegration:
    """Test end-to-end integration."""

    @patch("export_langsmith_traces.Client")
    @patch("sys.argv")
    def test_main_success_workflow(self, mock_argv, mock_client_class):
        """Test successful end-to-end execution of main()."""
        import tempfile
        import os
        from export_langsmith_traces import main

        # Arrange - Set up command-line arguments
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            temp_output = temp_file.name

        try:
            mock_argv.__getitem__ = Mock(
                side_effect=lambda i: [
                    "export_langsmith_traces.py",
                    "--api-key",
                    "test_key",
                    "--project",
                    "test-project",
                    "--limit",
                    "10",
                    "--output",
                    temp_output,
                ][i]
            )

            # Mock Client and its behavior
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Create mock runs with only essential attributes
            mock_run = Mock(spec=["id", "name"])
            mock_run.id = "run_123"
            mock_run.name = "test_run"

            mock_client.list_runs.return_value = [mock_run]

            # Act - Run main()
            main()

            # Assert - Verify file was created and contains data
            assert os.path.exists(temp_output)

            with open(temp_output, "r") as f:
                import json

                data = json.load(f)

            assert "export_metadata" in data
            assert "traces" in data
            assert data["export_metadata"]["total_traces"] == 1
            assert len(data["traces"]) == 1

        finally:
            # Cleanup
            if os.path.exists(temp_output):
                os.remove(temp_output)

    @patch("export_langsmith_traces.Client")
    @patch("sys.argv")
    def test_main_authentication_error(self, mock_argv, mock_client_class):
        """Test main() handles authentication error gracefully."""
        from export_langsmith_traces import main

        # Arrange
        mock_argv.__getitem__ = Mock(
            side_effect=lambda i: [
                "export_langsmith_traces.py",
                "--api-key",
                "invalid_key",
                "--project",
                "test-project",
                "--limit",
                "10",
                "--output",
                "output.json",
            ][i]
        )

        # Mock Client to raise authentication error
        mock_client_class.side_effect = Exception("Invalid API key")

        # Act & Assert - Should exit with status 1
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("export_langsmith_traces.Client")
    @patch("sys.argv")
    @patch("export_langsmith_traces.time.sleep")
    def test_main_with_pagination_success(
        self, mock_sleep, mock_argv, mock_client_class
    ):
        """Test full workflow with pagination (limit > 100)."""
        import tempfile
        import os
        import json
        from export_langsmith_traces import main

        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_output = f.name

        try:
            mock_argv.__getitem__ = Mock(
                side_effect=lambda i: [
                    "export_langsmith_traces.py",
                    "--api-key",
                    "test_key",
                    "--project",
                    "test-project",
                    "--limit",
                    "150",
                    "--output",
                    temp_output,
                ][i]
            )

            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Create 150 mock runs with proper attributes (not nested Mocks)
            all_runs = []
            for i in range(150):
                run = Mock()
                run.id = f"run_{i}"
                run.name = f"test_{i}"
                run.start_time = None
                run.end_time = None
                run.status = "completed"
                run.inputs = {}
                run.outputs = {}
                run.error = None
                run.run_type = "chain"
                run.child_runs = []
                all_runs.append(run)

            def mock_list_runs(*args, **kwargs):
                limit = kwargs.get("limit", len(all_runs))
                return iter(all_runs[:limit])

            mock_client.list_runs.side_effect = mock_list_runs

            # Act
            main()

            # Assert
            assert os.path.exists(temp_output)
            with open(temp_output, "r") as f:
                data = json.load(f)

            assert data["export_metadata"]["total_traces"] == 150
            assert len(data["traces"]) == 150

            # Verify multiple API calls were made (pagination)
            assert mock_client.list_runs.call_count == 2  # 150 records = 2 pages

        finally:
            if os.path.exists(temp_output):
                os.remove(temp_output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
