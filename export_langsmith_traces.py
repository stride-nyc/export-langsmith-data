#!/usr/bin/env python3
"""
LangSmith Data Export Script

Purpose: Export workflow trace data from LangSmith project for offline analysis.

Usage:
    python export_langsmith_traces.py \
        --api-key "lsv2_pt_..." \
        --project "project-name" \
        --limit 150 \
        --output "traces_export.json"

Requirements:
    - LangSmith API key (Individual Developer plan or higher)
    - Python 3.8+
    - Dependencies: langsmith, tqdm (optional)

Author: Generated with Claude Code (PDCA Framework)
Date: 2025-11-28
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from langsmith import Client


class AuthenticationError(Exception):
    """Raised when LangSmith API authentication fails."""

    pass


class ExportError(Exception):
    """Raised when JSON export fails."""

    pass


class ProjectNotFoundError(Exception):
    """Raised when LangSmith project doesn't exist."""

    pass


class RateLimitError(Exception):
    """Raised when rate limit exceeded after retries."""

    pass


class LangSmithExporter:
    """Handles LangSmith trace data export with rate limiting and error handling."""

    # API constants
    DEFAULT_API_URL = "https://api.smith.langchain.com"

    # Rate limiting constants
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 1.0  # seconds
    BACKOFF_MULTIPLIER = 2.0

    def __init__(self, api_key: str, api_url: str = DEFAULT_API_URL) -> None:
        """
        Initialize LangSmith client.

        Args:
            api_key: LangSmith API key for authentication
            api_url: LangSmith API endpoint URL

        Raises:
            AuthenticationError: If API key is invalid
        """
        self.api_key = api_key
        self.api_url = api_url

        try:
            self.client = Client(api_key=api_key, api_url=api_url)
        except Exception as e:
            raise AuthenticationError(
                f"Failed to authenticate with LangSmith API. "
                f"Please verify your API key is valid. Error: {str(e)}"
            ) from e

    def fetch_runs(self, project_name: str, limit: int) -> List[Any]:
        """
        Fetch runs from LangSmith with rate limiting.

        Args:
            project_name: Name of the LangSmith project
            limit: Maximum number of runs to retrieve

        Returns:
            List of Run objects from LangSmith

        Raises:
            ProjectNotFoundError: If project doesn't exist
            RateLimitError: If rate limit exceeded after retries
        """
        attempt = 0
        while attempt < self.MAX_RETRIES:
            try:
                runs = list(
                    self.client.list_runs(project_name=project_name, limit=limit)
                )
                return runs
            except Exception:
                attempt += 1
                if attempt >= self.MAX_RETRIES:
                    raise
                # Exponential backoff
                backoff_time = self.INITIAL_BACKOFF * (
                    self.BACKOFF_MULTIPLIER ** (attempt - 1)
                )
                time.sleep(backoff_time)
        # This should never be reached, but mypy needs it
        raise Exception("Max retries exceeded")

    def format_trace_data(self, runs: List[Any]) -> Dict[str, Any]:
        """
        Transform Run objects to output JSON schema.

        Args:
            runs: List of LangSmith Run objects

        Returns:
            Dictionary matching the export schema
        """
        # Create metadata
        export_metadata = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_traces": len(runs),
            "langsmith_api_version": "0.4.x",
        }

        # Transform runs to trace format
        traces = []
        for run in runs:
            # Calculate duration
            duration_seconds = 0
            if hasattr(run, "start_time") and hasattr(run, "end_time"):
                if run.start_time and run.end_time:
                    duration_seconds = (run.end_time - run.start_time).total_seconds()

            trace = {
                "id": str(getattr(run, "id", None)) if hasattr(run, "id") else None,
                "name": getattr(run, "name", None),
                "start_time": (
                    run.start_time.isoformat()
                    if hasattr(run, "start_time") and run.start_time
                    else None
                ),
                "end_time": (
                    run.end_time.isoformat()
                    if hasattr(run, "end_time") and run.end_time
                    else None
                ),
                "duration_seconds": duration_seconds,
                "status": getattr(run, "status", None),
                "inputs": getattr(run, "inputs", {}),
                "outputs": getattr(run, "outputs", {}),
                "error": getattr(run, "error", None),
                "run_type": getattr(run, "run_type", None),
                "child_runs": getattr(run, "child_runs", []),
            }
            traces.append(trace)

        return {"export_metadata": export_metadata, "traces": traces}

    def export_to_json(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Save formatted data to JSON file.

        Args:
            data: Formatted trace data dictionary
            filepath: Output file path

        Raises:
            ExportError: If file write fails
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ExportError(
                f"Failed to export data to {filepath}. Error: {str(e)}"
            ) from e

    def _handle_rate_limit(self, attempt: int) -> None:
        """
        Implement exponential backoff for rate limiting.

        Args:
            attempt: Current retry attempt number
        """
        pass


def _positive_int(value: str) -> int:
    """
    Validate that argument is a positive integer.

    Args:
        value: String value from command line

    Returns:
        Integer value if valid

    Raises:
        argparse.ArgumentTypeError: If value is not a positive integer
    """
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                f"{value} must be a positive integer (> 0)"
            )
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} must be an integer")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Export LangSmith trace data for offline analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python export_langsmith_traces.py \\
      --api-key "lsv2_pt_..." \\
      --project "my-project" \\
      --limit 150 \\
      --output "traces_export.json"
        """,
    )

    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="LangSmith API key for authentication",
    )

    parser.add_argument(
        "--project", type=str, required=True, help="LangSmith project name or ID"
    )

    parser.add_argument(
        "--limit",
        type=_positive_int,
        required=True,
        help="Number of most recent traces to export (must be > 0)",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Output JSON file path"
    )

    return parser.parse_args()


def main() -> None:
    """
    Main execution function that orchestrates the export workflow.

    Workflow:
    1. Parse command-line arguments
    2. Initialize LangSmith client
    3. Fetch runs from project
    4. Format trace data
    5. Export to JSON file

    Exits with status code 0 on success, 1 on error.
    """
    # Ensure UTF-8 encoding for console output (Windows compatibility)
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except AttributeError:
            # Python < 3.7 doesn't have reconfigure
            pass

    try:
        # Parse arguments
        args = parse_arguments()

        print(f"🚀 Exporting {args.limit} traces from project '{args.project}'...")

        # Initialize exporter
        try:
            exporter = LangSmithExporter(api_key=args.api_key)
            print("✓ Connected to LangSmith API")
        except AuthenticationError as e:
            print(f"❌ Authentication failed: {e}", file=sys.stderr)
            sys.exit(1)

        # Fetch runs
        try:
            print("📥 Fetching traces...")
            runs = exporter.fetch_runs(project_name=args.project, limit=args.limit)
            print(f"✓ Fetched {len(runs)} traces")

            if len(runs) == 0:
                print("⚠️  No traces found in project")
                # Still export empty result
        except Exception as e:
            print(f"❌ Failed to fetch traces: {e}", file=sys.stderr)
            sys.exit(1)

        # Format data
        print("🔄 Formatting trace data...")
        formatted_data = exporter.format_trace_data(runs)
        print("✓ Data formatted")

        # Export to JSON
        try:
            print(f"💾 Exporting to {args.output}...")
            exporter.export_to_json(formatted_data, args.output)
            print(f"✅ Export complete! Saved to {args.output}")
        except ExportError as e:
            print(f"❌ Failed to export data: {e}", file=sys.stderr)
            sys.exit(1)

        # Success summary
        print("\n📊 Summary:")
        print(f"   Total traces exported: {len(runs)}")
        print(f"   Output file: {args.output}")

    except KeyboardInterrupt:
        print("\n⚠️  Export cancelled by user", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
