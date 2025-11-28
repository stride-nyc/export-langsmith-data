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

    def test_format_trace_data_complete_fields(self):
        """Test data formatting with all fields present."""
        pass

    def test_format_trace_data_missing_fields(self):
        """Test safe handling of missing/null fields."""
        pass

    def test_format_trace_data_with_child_runs(self):
        """Test nested run relationship handling."""
        pass

    def test_export_to_json_success(self):
        """Test successful JSON file creation."""
        pass

    def test_export_to_json_write_error(self):
        """Test file write error handling."""
        pass


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
