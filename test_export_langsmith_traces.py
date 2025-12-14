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

    def test_parse_arguments_include_children_defaults_to_false(self):
        """Test that --include-children flag defaults to False when not specified."""
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
        assert args.include_children is False

    def test_parse_arguments_include_children_flag(self):
        """Test that --include-children flag is parsed correctly."""
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
            "--include-children",
        ]

        # Act
        with patch("sys.argv", ["export_langsmith_traces.py"] + test_args):
            args = parse_arguments()

        # Assert
        assert args.include_children is True


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
        # Updated: no limit passed to API (SDK handles internal pagination)
        mock_client.list_runs.assert_called_once_with(project_name="test-project")
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
    def test_format_trace_data_includes_token_usage(self, mock_client_class):
        """Test that token usage fields are exported when present."""
        # Arrange
        from datetime import datetime, timezone

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create mock LLM Run with token usage
        mock_run = Mock()
        mock_run.id = "llm_run_456"
        mock_run.name = "ChatGoogleGenerativeAI"
        mock_run.start_time = datetime(2025, 12, 9, 10, 0, 0, tzinfo=timezone.utc)
        mock_run.end_time = datetime(2025, 12, 9, 10, 2, 0, tzinfo=timezone.utc)
        mock_run.status = "success"
        mock_run.inputs = {"messages": []}
        mock_run.outputs = {"generations": []}
        mock_run.error = None
        mock_run.run_type = "llm"
        mock_run.child_runs = []
        # Token usage fields (as found in LangSmith API)
        mock_run.total_tokens = 162286
        mock_run.prompt_tokens = 128227
        mock_run.completion_tokens = 34059

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        result = exporter.format_trace_data([mock_run])

        # Assert
        trace = result["traces"][0]
        assert "total_tokens" in trace
        assert "prompt_tokens" in trace
        assert "completion_tokens" in trace
        assert trace["total_tokens"] == 162286
        assert trace["prompt_tokens"] == 128227
        assert trace["completion_tokens"] == 34059

    @patch("export_langsmith_traces.Client")
    def test_format_trace_data_handles_missing_tokens(self, mock_client_class):
        """Test that missing token fields are handled gracefully."""
        # Arrange
        from datetime import datetime, timezone

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create mock non-LLM Run without token usage
        mock_run = Mock()
        mock_run.id = "chain_run_789"
        mock_run.name = "LangGraph"
        mock_run.start_time = datetime(2025, 12, 9, 10, 0, 0, tzinfo=timezone.utc)
        mock_run.end_time = datetime(2025, 12, 9, 10, 5, 0, tzinfo=timezone.utc)
        mock_run.status = "success"
        mock_run.inputs = {}
        mock_run.outputs = {}
        mock_run.error = None
        mock_run.run_type = "chain"
        mock_run.child_runs = []
        # No token fields (non-LLM run)
        mock_run.total_tokens = None
        mock_run.prompt_tokens = None
        mock_run.completion_tokens = None

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        result = exporter.format_trace_data([mock_run])

        # Assert
        trace = result["traces"][0]
        # Token fields should be present but None for non-LLM runs
        assert "total_tokens" in trace
        assert "prompt_tokens" in trace
        assert "completion_tokens" in trace
        assert trace["total_tokens"] is None
        assert trace["prompt_tokens"] is None
        assert trace["completion_tokens"] is None

    @patch("export_langsmith_traces.Client")
    def test_format_trace_data_includes_cache_tokens_from_nested_fields(
        self, mock_client_class
    ):
        """Test that cache token fields are extracted from nested input_token_details (LangSmith API structure)."""
        # Arrange
        from datetime import datetime, timezone

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create mock LLM Run with cache token usage in nested input_token_details
        # Use spec to prevent Mock from creating cache_read_tokens/cache_creation_tokens automatically
        mock_run = Mock(
            spec=[
                "id",
                "name",
                "start_time",
                "end_time",
                "status",
                "inputs",
                "outputs",
                "error",
                "run_type",
                "child_runs",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
            ]
        )
        mock_run.id = "cached_llm_run_999"
        mock_run.name = "ChatGoogleGenerativeAI"
        mock_run.start_time = datetime(2025, 12, 11, 10, 0, 0, tzinfo=timezone.utc)
        mock_run.end_time = datetime(2025, 12, 11, 10, 2, 0, tzinfo=timezone.utc)
        mock_run.status = "success"
        mock_run.inputs = {"messages": []}
        mock_run.error = None
        mock_run.run_type = "llm"
        mock_run.child_runs = []
        # Standard token fields
        mock_run.total_tokens = 50000
        mock_run.prompt_tokens = 10000
        mock_run.completion_tokens = 5000
        # Cache token fields in nested LangChain message structure (actual LangSmith export format)
        # These are nested under outputs["generations"][0][0]["message"]["kwargs"]["usage_metadata"]["input_token_details"]
        mock_run.outputs = {
            "generations": [
                [
                    {
                        "message": {
                            "kwargs": {
                                "usage_metadata": {
                                    "input_tokens": 10000,
                                    "output_tokens": 5000,
                                    "total_tokens": 50000,
                                    "input_token_details": {
                                        "cache_read": 35000,  # Tokens read from cache
                                        "cache_creation": 0,  # Tokens written to cache
                                    },
                                }
                            }
                        }
                    }
                ]
            ]
        }
        # Top-level cache fields not present (not in spec, so getattr will return None)

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        result = exporter.format_trace_data([mock_run])

        # Assert
        trace = result["traces"][0]
        assert "cache_read_tokens" in trace
        assert "cache_creation_tokens" in trace
        assert trace["cache_read_tokens"] == 35000
        assert trace["cache_creation_tokens"] == 0
        # Standard tokens should still be present
        assert trace["total_tokens"] == 50000
        assert trace["prompt_tokens"] == 10000
        assert trace["completion_tokens"] == 5000

    @patch("export_langsmith_traces.Client")
    def test_format_trace_data_handles_missing_cache_tokens(self, mock_client_class):
        """Test that missing cache token fields are handled gracefully (older LangSmith data)."""
        # Arrange
        from datetime import datetime, timezone

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create mock LLM Run WITHOUT cache token fields (older export or non-cached run)
        mock_run = Mock(
            spec=[
                "id",
                "name",
                "start_time",
                "end_time",
                "status",
                "inputs",
                "outputs",
                "error",
                "run_type",
                "child_runs",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
            ]
        )
        mock_run.id = "old_llm_run_888"
        mock_run.name = "ChatGoogleGenerativeAI"
        mock_run.start_time = datetime(2025, 12, 11, 10, 0, 0, tzinfo=timezone.utc)
        mock_run.end_time = datetime(2025, 12, 11, 10, 2, 0, tzinfo=timezone.utc)
        mock_run.status = "success"
        mock_run.inputs = {"messages": []}
        mock_run.outputs = {"generations": []}
        mock_run.error = None
        mock_run.run_type = "llm"
        mock_run.child_runs = []
        # Standard token fields present
        mock_run.total_tokens = 15000
        mock_run.prompt_tokens = 10000
        mock_run.completion_tokens = 5000
        # Cache token fields NOT present (not in spec, so getattr will return None via default)

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        result = exporter.format_trace_data([mock_run])

        # Assert
        trace = result["traces"][0]
        # Cache token fields should be present but None when not available in source data
        assert "cache_read_tokens" in trace
        assert "cache_creation_tokens" in trace
        assert trace["cache_read_tokens"] is None
        assert trace["cache_creation_tokens"] is None
        # Standard tokens should still be present
        assert trace["total_tokens"] == 15000
        assert trace["prompt_tokens"] == 10000
        assert trace["completion_tokens"] == 5000

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
        # Child runs are now formatted as dictionaries (not Mock objects)
        assert parent_trace["child_runs"][0]["id"] == "child_1"
        assert parent_trace["child_runs"][0]["name"] == "child_workflow_1"
        assert parent_trace["child_runs"][1]["id"] == "child_2"
        assert parent_trace["child_runs"][1]["name"] == "child_workflow_2"

    @patch("export_langsmith_traces.Client")
    def test_format_trace_data_recursively_formats_children(self, mock_client_class):
        """Test that child_runs are recursively formatted as dictionaries, not Run objects."""
        import json
        import tempfile
        import os

        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create a child run with all attributes
        child_run = Mock(
            spec=[
                "id",
                "name",
                "start_time",
                "end_time",
                "status",
                "inputs",
                "outputs",
                "error",
                "run_type",
                "child_runs",
            ]
        )
        child_run.id = "child_1"
        child_run.name = "child_workflow"
        child_run.start_time = None
        child_run.end_time = None
        child_run.status = "completed"
        child_run.inputs = {}
        child_run.outputs = {}
        child_run.error = None
        child_run.run_type = "chain"
        child_run.child_runs = []

        # Create parent run with child
        parent_run = Mock(
            spec=[
                "id",
                "name",
                "start_time",
                "end_time",
                "status",
                "inputs",
                "outputs",
                "error",
                "run_type",
                "child_runs",
            ]
        )
        parent_run.id = "parent_1"
        parent_run.name = "parent_workflow"
        parent_run.start_time = None
        parent_run.end_time = None
        parent_run.status = "completed"
        parent_run.inputs = {}
        parent_run.outputs = {}
        parent_run.error = None
        parent_run.run_type = "chain"
        parent_run.child_runs = [child_run]

        exporter = LangSmithExporter(api_key="test_key")

        # Act
        formatted_data = exporter.format_trace_data([parent_run])

        # Assert - formatted data should be JSON serializable
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # This should not raise an exception
            with open(temp_path, "w") as f:
                json.dump(formatted_data, f)

            # Verify the structure
            with open(temp_path, "r") as f:
                data = json.load(f)

            assert len(data["traces"]) == 1
            parent_trace = data["traces"][0]
            assert parent_trace["id"] == "parent_1"
            assert len(parent_trace["child_runs"]) == 1

            # Child should be a dictionary, not a Mock object
            child_trace = parent_trace["child_runs"][0]
            assert isinstance(child_trace, dict)
            assert child_trace["id"] == "child_1"
            assert child_trace["name"] == "child_workflow"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

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
        # Updated: no limit passed to API (SDK handles internal pagination)
        mock_client.list_runs.assert_called_with(project_name="test-project")

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

        # Updated: Verify no limit parameter passed (SDK handles internal pagination)
        calls = mock_client.list_runs.call_args_list
        for call in calls:
            assert "limit" not in call.kwargs, "No limit should be passed to API"

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

    @patch("export_langsmith_traces.Client")
    @patch("export_langsmith_traces.time.sleep")
    def test_fetch_runs_respects_api_100_limit(self, mock_sleep, mock_client_class):
        """Test that pagination never passes limit > 100 to API (real API constraint)."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create 250 mock runs
        all_mock_runs = [Mock(id=f"run_{i}", name=f"test_{i}") for i in range(250)]

        # Mock list_runs to REJECT limit > 100 (simulating real API behavior)
        # When no limit specified, return ALL runs (simulating SDK's internal pagination)
        def mock_list_runs(*args, **kwargs):
            limit = kwargs.get("limit", None)
            if limit is not None and limit > 100:
                raise Exception(
                    "Failed to POST /runs/query in LangSmith API. "
                    "HTTPError('400 Client Error: Bad Request', "
                    '\'{"detail":"Limit exceeds maximum allowed value of 100"}\')'
                )
            # If no limit specified, return all runs (SDK handles internal pagination)
            if limit is None:
                return iter(all_mock_runs)
            else:
                return iter(all_mock_runs[:limit])

        mock_client.list_runs.side_effect = mock_list_runs

        exporter = LangSmithExporter(api_key="test_key")

        # Act - Request 250 runs (should handle pagination internally)
        runs = exporter.fetch_runs(project_name="test-project", limit=250)

        # Assert
        assert len(runs) == 250
        # Verify NO call passed limit > 100 to the API
        # (None or missing limit is acceptable - SDK handles pagination internally)
        for call in mock_client.list_runs.call_args_list:
            limit_arg = call.kwargs.get("limit", None)
            assert (
                limit_arg is None or limit_arg <= 100
            ), f"API called with limit={limit_arg} (must be None or <= 100)"

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


class TestHierarchicalDataFetching:
    """Test fetching runs with child relationships."""

    @patch("export_langsmith_traces.Client")
    def test_fetch_runs_with_children_single_run(self, mock_client_class):
        """Test fetching a single run with its children."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        exporter = LangSmithExporter(api_key="test_key")
        exporter.client = mock_client

        # Mock a parent run from list_runs (without children)
        parent_run = Mock()
        parent_run.id = "parent-123"
        parent_run.name = "LangGraph"
        parent_run.run_type = "chain"
        parent_run.child_runs = None  # list_runs doesn't populate this

        # Mock the full run with children from read_run
        child1 = Mock()
        child1.id = "child-1"
        child1.name = "process_data"
        child1.run_type = "chain"
        child1.child_runs = []

        child2 = Mock()
        child2.id = "child-2"
        child2.name = "transform_output"
        child2.run_type = "chain"
        child2.child_runs = []

        full_parent_run = Mock()
        full_parent_run.id = "parent-123"
        full_parent_run.name = "LangGraph"
        full_parent_run.run_type = "chain"
        full_parent_run.child_runs = [child1, child2]

        # Configure mocks
        mock_client.list_runs.return_value = iter([parent_run])
        mock_client.read_run.return_value = full_parent_run

        # Act
        runs = exporter.fetch_runs_with_children(project_name="test-project", limit=1)

        # Assert
        assert len(runs) == 1
        assert runs[0].id == "parent-123"
        assert runs[0].child_runs is not None
        assert len(runs[0].child_runs) == 2
        assert runs[0].child_runs[0].name == "process_data"
        assert runs[0].child_runs[1].name == "transform_output"

        # Verify read_run was called with load_child_runs=True
        mock_client.read_run.assert_called_once_with("parent-123", load_child_runs=True)

    @patch("export_langsmith_traces.Client")
    def test_fetch_runs_with_children_multiple_runs(self, mock_client_class):
        """Test fetching multiple runs with their children."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        exporter = LangSmithExporter(api_key="test_key")
        exporter.client = mock_client

        # Mock 3 parent runs from list_runs
        parent1 = Mock(id="p1", name="LangGraph", child_runs=None)
        parent2 = Mock(id="p2", name="LangGraph", child_runs=None)
        parent3 = Mock(id="p3", name="LangGraph", child_runs=None)

        # Mock full runs with children
        full1 = Mock(id="p1", name="LangGraph", child_runs=[Mock(name="node1")])
        full2 = Mock(id="p2", name="LangGraph", child_runs=[Mock(name="node2")])
        full3 = Mock(id="p3", name="LangGraph", child_runs=[Mock(name="node3")])

        mock_client.list_runs.return_value = iter([parent1, parent2, parent3])
        mock_client.read_run.side_effect = [full1, full2, full3]

        # Act
        runs = exporter.fetch_runs_with_children(project_name="test-project", limit=3)

        # Assert
        assert len(runs) == 3
        assert all(run.child_runs is not None for run in runs)
        assert mock_client.read_run.call_count == 3

    @patch("export_langsmith_traces.Client")
    def test_fetch_runs_with_children_handles_api_error(self, mock_client_class):
        """Test that fetch_runs_with_children gracefully handles errors."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        exporter = LangSmithExporter(api_key="test_key")
        exporter.client = mock_client

        parent = Mock(id="p1", name="LangGraph", child_runs=None)
        mock_client.list_runs.return_value = iter([parent])
        mock_client.read_run.side_effect = Exception("API Error")

        # Act
        runs = exporter.fetch_runs_with_children(project_name="test-project", limit=1)

        # Assert - should fall back to flat run
        assert len(runs) == 1
        assert runs[0].id == "p1"  # Falls back to original run

    @patch("export_langsmith_traces.Client")
    def test_fetch_runs_with_children_empty_project(self, mock_client_class):
        """Test fetch_runs_with_children with empty project."""
        # Arrange
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        exporter = LangSmithExporter(api_key="test_key")
        exporter.client = mock_client
        mock_client.list_runs.return_value = iter([])

        # Act
        runs = exporter.fetch_runs_with_children(project_name="test-project", limit=10)

        # Assert
        assert len(runs) == 0
        mock_client.read_run.assert_not_called()


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
            mock_run = Mock(
                spec=[
                    "id",
                    "name",
                    "start_time",
                    "end_time",
                    "status",
                    "inputs",
                    "outputs",
                    "error",
                    "run_type",
                    "child_runs",
                    "total_tokens",
                    "prompt_tokens",
                    "completion_tokens",
                ]
            )
            mock_run.id = "run_123"
            mock_run.name = "test_run"
            mock_run.start_time = None
            mock_run.end_time = None
            mock_run.status = "completed"
            mock_run.inputs = {}
            mock_run.outputs = {}
            mock_run.error = None
            mock_run.run_type = "chain"
            mock_run.child_runs = []
            mock_run.total_tokens = None
            mock_run.prompt_tokens = None
            mock_run.completion_tokens = None

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
                # Token fields (None for non-LLM runs)
                run.total_tokens = None
                run.prompt_tokens = None
                run.completion_tokens = None
                # Cache token fields (None for non-cached runs)
                run.cache_read_tokens = None
                run.cache_creation_tokens = None
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

    @patch("export_langsmith_traces.Client")
    @patch("sys.argv")
    @patch("export_langsmith_traces.time.sleep")  # Speed up test
    def test_main_with_include_children_flag(
        self, mock_sleep, mock_argv, mock_client_class
    ):
        """Test that main() uses fetch_runs_with_children() when --include-children is set."""
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
                    "2",
                    "--output",
                    temp_output,
                    "--include-children",
                ][i]
            )

            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Create mock flat runs (from list_runs)
            flat_run1 = Mock()
            flat_run1.id = "run_1"
            flat_run1.name = "LangGraph"
            flat_run1.child_runs = None  # list_runs doesn't populate this

            flat_run2 = Mock()
            flat_run2.id = "run_2"
            flat_run2.name = "LangGraph"
            flat_run2.child_runs = None

            # Create mock full runs with children (from read_run)
            # Use simple Mock spec to avoid JSON serialization issues
            full_run1 = Mock(
                spec=[
                    "id",
                    "name",
                    "start_time",
                    "end_time",
                    "status",
                    "inputs",
                    "outputs",
                    "error",
                    "run_type",
                    "child_runs",
                    "total_tokens",
                    "prompt_tokens",
                    "completion_tokens",
                ]
            )
            full_run1.id = "run_1"
            full_run1.name = "LangGraph"
            full_run1.start_time = None
            full_run1.end_time = None
            full_run1.status = "completed"
            full_run1.inputs = {}
            full_run1.outputs = {}
            full_run1.error = None
            full_run1.run_type = "chain"
            full_run1.child_runs = []  # Empty list to avoid nested Mock serialization
            full_run1.total_tokens = None
            full_run1.prompt_tokens = None
            full_run1.completion_tokens = None

            full_run2 = Mock(
                spec=[
                    "id",
                    "name",
                    "start_time",
                    "end_time",
                    "status",
                    "inputs",
                    "outputs",
                    "error",
                    "run_type",
                    "child_runs",
                    "total_tokens",
                    "prompt_tokens",
                    "completion_tokens",
                ]
            )
            full_run2.id = "run_2"
            full_run2.name = "LangGraph"
            full_run2.start_time = None
            full_run2.end_time = None
            full_run2.status = "completed"
            full_run2.inputs = {}
            full_run2.outputs = {}
            full_run2.error = None
            full_run2.run_type = "chain"
            full_run2.child_runs = []
            full_run2.total_tokens = None
            full_run2.prompt_tokens = None
            full_run2.completion_tokens = None

            # Mock behaviors
            mock_client.list_runs.return_value = iter([flat_run1, flat_run2])
            mock_client.read_run.side_effect = [full_run1, full_run2]

            # Act
            main()

            # Assert
            assert os.path.exists(temp_output)

            # Verify read_run was called (indicating hierarchical fetch was used)
            assert mock_client.read_run.call_count == 2
            mock_client.read_run.assert_any_call("run_1", load_child_runs=True)
            mock_client.read_run.assert_any_call("run_2", load_child_runs=True)

            # Verify the output file was created successfully
            with open(temp_output, "r") as f:
                data = json.load(f)

            assert "traces" in data
            assert len(data["traces"]) == 2
            # The key test is that read_run was called with load_child_runs=True
            # In real usage, the SDK will return proper Run objects with populated children

        finally:
            if os.path.exists(temp_output):
                os.remove(temp_output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
