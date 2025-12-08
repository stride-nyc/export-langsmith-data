"""
Tests for validate_export.py

Test coverage for the export validation utility.
"""

import json
import sys
from unittest.mock import patch

import pytest

from validate_export import main, validate_export_file


class TestValidateExportFile:
    """Tests for validate_export_file function."""

    def test_file_not_found(self, tmp_path, capsys):
        """Test validation of non-existent file."""
        non_existent = tmp_path / "does_not_exist.json"

        with pytest.raises(SystemExit) as exc_info:
            validate_export_file(str(non_existent))

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: File not found" in captured.out

    def test_invalid_json_file(self, tmp_path, capsys):
        """Test validation of invalid JSON file."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json {")

        with pytest.raises(SystemExit) as exc_info:
            validate_export_file(str(invalid_file))

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error loading file" in captured.out

    def test_valid_export_with_validators(self, tmp_path, capsys):
        """Test validation of valid export with validator workflows."""
        # Create valid export file
        export_data = {
            "export_metadata": {
                "export_timestamp": "2025-01-01T00:00:00Z",
                "project_name": "test-project",
                "total_traces": 2,
            },
            "traces": [
                {
                    "id": "workflow_1",
                    "name": "LangGraph",
                    "start_time": "2025-01-01T00:00:00Z",
                    "end_time": "2025-01-01T00:15:00Z",
                    "duration_seconds": 900,
                    "status": "success",
                    "run_type": "chain",
                    "child_runs": [
                        {
                            "id": "child_1",
                            "name": "meta_evaluation",
                            "start_time": "2025-01-01T00:00:00Z",
                            "end_time": "2025-01-01T00:05:00Z",
                            "duration_seconds": 300,
                            "status": "success",
                            "run_type": "chain",
                        },
                        {
                            "id": "child_2",
                            "name": "normative_validation",
                            "start_time": "2025-01-01T00:00:00Z",
                            "end_time": "2025-01-01T00:05:00Z",
                            "duration_seconds": 300,
                            "status": "success",
                            "run_type": "chain",
                        },
                        {
                            "id": "child_3",
                            "name": "simulated_testing",
                            "start_time": "2025-01-01T00:00:00Z",
                            "end_time": "2025-01-01T00:05:00Z",
                            "duration_seconds": 300,
                            "status": "success",
                            "run_type": "chain",
                        },
                    ],
                },
                {
                    "id": "orphan_1",
                    "name": "standalone_run",
                    "start_time": "2025-01-01T00:00:00Z",
                    "end_time": "2025-01-01T00:05:00Z",
                    "duration_seconds": 300,
                    "status": "success",
                    "run_type": "llm",
                },
            ],
        }

        export_file = tmp_path / "valid_export.json"
        export_file.write_text(json.dumps(export_data))

        # Run validation
        validate_export_file(str(export_file))

        # Check output
        captured = capsys.readouterr()
        assert "Data loaded successfully!" in captured.out
        assert "Total workflows:        1" in captured.out
        assert "Orphan traces:          1" in captured.out
        assert "Hierarchical data:      Yes" in captured.out
        assert "Workflows with all validators: 1" in captured.out

    def test_valid_export_without_validators(self, tmp_path, capsys):
        """Test validation of export without validator workflows."""
        export_data = {
            "export_metadata": {
                "export_timestamp": "2025-01-01T00:00:00Z",
                "project_name": "test-project",
                "total_traces": 1,
            },
            "traces": [
                {
                    "id": "workflow_1",
                    "name": "LangGraph",
                    "start_time": "2025-01-01T00:00:00Z",
                    "end_time": "2025-01-01T00:15:00Z",
                    "duration_seconds": 900,
                    "status": "success",
                    "run_type": "chain",
                    "child_runs": [
                        {
                            "id": "child_1",
                            "name": "some_other_node",
                            "start_time": "2025-01-01T00:00:00Z",
                            "end_time": "2025-01-01T00:05:00Z",
                            "duration_seconds": 300,
                            "status": "success",
                            "run_type": "chain",
                        }
                    ],
                }
            ],
        }

        export_file = tmp_path / "no_validators.json"
        export_file.write_text(json.dumps(export_data))

        validate_export_file(str(export_file))

        captured = capsys.readouterr()
        assert "Workflows with any validator:  0" in captured.out
        assert "Workflows with all validators: 0" in captured.out
        assert "Parallel Analysis:      INSUFFICIENT" in captured.out

    def test_large_dataset_validation(self, tmp_path, capsys):
        """Test validation shows EXCELLENT for large datasets."""
        # Create large dataset (150 workflows with child_runs to be recognized as workflows)
        traces = []
        for i in range(150):
            traces.append(
                {
                    "id": f"workflow_{i}",
                    "name": "LangGraph",
                    "start_time": "2025-01-01T00:00:00Z",
                    "end_time": "2025-01-01T00:15:00Z",
                    "duration_seconds": 900,
                    "status": "success",
                    "run_type": "chain",
                    "child_runs": [
                        {
                            "id": f"child_{i}",
                            "name": "some_node",
                            "start_time": "2025-01-01T00:00:00Z",
                            "end_time": "2025-01-01T00:05:00Z",
                            "duration_seconds": 300,
                            "status": "success",
                            "run_type": "chain",
                        }
                    ],
                }
            )

        export_data = {
            "export_metadata": {
                "export_timestamp": "2025-01-01T00:00:00Z",
                "project_name": "test-project",
                "total_traces": 150,
            },
            "traces": traces,
        }

        export_file = tmp_path / "large_export.json"
        export_file.write_text(json.dumps(export_data))

        validate_export_file(str(export_file))

        captured = capsys.readouterr()
        assert "Total workflows:        150" in captured.out
        assert "Latency Analysis:       EXCELLENT" in captured.out
        assert "Bottleneck Analysis:    EXCELLENT" in captured.out


class TestMainFunction:
    """Tests for main CLI function."""

    def test_main_no_arguments(self, capsys):
        """Test main function with no arguments."""
        with patch.object(sys, "argv", ["validate_export.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Usage:" in captured.out

    def test_main_with_file_argument(self, tmp_path, capsys):
        """Test main function with file argument."""
        # Create minimal valid export
        export_data = {
            "export_metadata": {
                "export_timestamp": "2025-01-01T00:00:00Z",
                "project_name": "test",
                "total_traces": 1,
            },
            "traces": [
                {
                    "id": "run_1",
                    "name": "test",
                    "start_time": "2025-01-01T00:00:00Z",
                    "end_time": "2025-01-01T00:01:00Z",
                    "duration_seconds": 60,
                    "status": "success",
                    "run_type": "llm",
                }
            ],
        }

        export_file = tmp_path / "test_export.json"
        export_file.write_text(json.dumps(export_data))

        with patch.object(sys, "argv", ["validate_export.py", str(export_file)]):
            main()

        captured = capsys.readouterr()
        assert "Data loaded successfully!" in captured.out
