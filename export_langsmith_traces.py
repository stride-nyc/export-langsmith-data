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
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
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

    def _looks_like_uuid(self, value: str) -> bool:
        """Check if a string looks like a UUID."""
        import re

        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        return bool(uuid_pattern.match(value))

    def fetch_runs(self, project_name: str, limit: int) -> List[Any]:
        """
        Fetch runs from LangSmith with pagination support for large exports.

        Due to LangSmith API limitations (max 100 records per call), this method
        makes multiple API calls to fetch all requested runs.

        Args:
            project_name: Name or ID of the LangSmith project
            limit: Maximum number of runs to retrieve

        Returns:
            List of Run objects from LangSmith

        Raises:
            ProjectNotFoundError: If project doesn't exist
            RateLimitError: If rate limit exceeded after retries
        """
        CHUNK_SIZE = 100  # LangSmith API limit per call

        all_runs = []
        fetched_count = 0

        # Calculate number of pages needed
        num_pages = (limit + CHUNK_SIZE - 1) // CHUNK_SIZE

        # Only show pagination message if multiple pages needed
        if num_pages > 1:
            print(f"  📄 Fetching {limit} runs across {num_pages} pages...")

        for page_num in range(num_pages):
            # Calculate how many runs to fetch in this page
            remaining = limit - fetched_count
            page_size = min(CHUNK_SIZE, remaining)

            # Fetch this page
            page_runs = self._fetch_page_with_retry(
                project_name=project_name,
                limit=page_size,
                fetched_so_far=fetched_count,
                page_num=page_num + 1,
                total_pages=num_pages,
            )

            # No more runs available
            if len(page_runs) == 0:
                if fetched_count == 0:
                    # No runs at all - will be handled by caller
                    break
                else:
                    # Got some runs but not all requested
                    print(
                        f"  ℹ️  Reached end of available runs at {fetched_count} (requested {limit})"
                    )
                    break

            all_runs.extend(page_runs)
            fetched_count += len(page_runs)

            # Progress update for multi-page fetches
            if num_pages > 1:
                print(
                    f"    ✓ Page {page_num + 1}/{num_pages}: {len(page_runs)} runs (Total: {fetched_count})"
                )

            # Check if we got fewer than requested - indicates no more runs available
            if len(page_runs) < page_size:
                if fetched_count < limit:
                    print(
                        f"  ℹ️  Only {fetched_count} runs available (requested {limit})"
                    )
                break

            # Reached our limit
            if fetched_count >= limit:
                break

            # Add small delay between pages (not on last page)
            if page_num < num_pages - 1 and fetched_count < limit:
                time.sleep(0.5)  # 500ms delay between pages

        # Final warning if significantly fewer runs than requested
        if fetched_count < limit:
            print(f"  ⚠️  Warning: Fetched {fetched_count} runs (requested {limit})")

        return all_runs

    def _fetch_page_with_retry(
        self,
        project_name: str,
        limit: int,
        fetched_so_far: int,
        page_num: int,
        total_pages: int,
    ) -> List[Any]:
        """
        Fetch a single page of runs with exponential backoff retry logic.

        This method wraps the SDK's list_runs call with retry logic to handle
        transient errors and rate limiting.

        Since LangSmith SDK doesn't support offset parameter, we request all runs
        up to our position + page size, then skip to our position using islice.

        Args:
            project_name: Name or ID of the LangSmith project
            limit: Number of runs to fetch for this page
            fetched_so_far: Number of runs already fetched (used for offset simulation)
            page_num: Current page number (1-indexed, for logging)
            total_pages: Total number of pages expected (for logging)

        Returns:
            List of Run objects from this page

        Raises:
            ProjectNotFoundError: If project doesn't exist
            RateLimitError: If rate limit exceeded after retries
        """
        from itertools import islice

        attempt = 0
        last_exception = None

        while attempt < self.MAX_RETRIES:
            try:
                # Since LangSmith SDK doesn't support offset parameter,
                # we request all runs up to our position + page size,
                # then skip to our position using islice
                total_to_request = fetched_so_far + limit

                # Try with project_name first
                runs_iterator = self.client.list_runs(
                    project_name=project_name, limit=total_to_request
                )

                # Skip already-fetched runs and take the next page
                page_runs = list(
                    islice(runs_iterator, fetched_so_far, fetched_so_far + limit)
                )

                return page_runs

            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check if this is a project not found error (not a rate limit or network error)
                if any(
                    term in error_msg
                    for term in ["not found", "does not exist", "project", "404"]
                ):
                    # If it looks like a UUID, try as project_id instead
                    if self._looks_like_uuid(project_name):
                        print("Trying project ID instead of name...")
                        try:
                            runs_iterator = self.client.list_runs(
                                project_id=project_name, limit=total_to_request
                            )
                            page_runs = list(
                                islice(
                                    runs_iterator,
                                    fetched_so_far,
                                    fetched_so_far + limit,
                                )
                            )
                            return page_runs
                        except Exception:  # nosec B110
                            pass  # Intentional: Fall through to retry logic if project_id also fails

                    # If first attempt and looks like project name issue, raise specific error
                    if attempt == 0:
                        raise ProjectNotFoundError(
                            f"Project '{project_name}' not found. "
                            f"Please verify the project name or try using the project ID (UUID format). "
                            f"You can find the project ID in the LangSmith URL when viewing your project. "
                            f"Original error: {str(e)}"
                        ) from e

                attempt += 1
                if attempt >= self.MAX_RETRIES:
                    break

                # Exponential backoff
                backoff_time = self.INITIAL_BACKOFF * (
                    self.BACKOFF_MULTIPLIER ** (attempt - 1)
                )

                # Only show retry message for multi-page fetches
                if total_pages > 1:
                    print(
                        f"    ⚠️  Page {page_num}/{total_pages} failed (attempt {attempt}/{self.MAX_RETRIES}), retrying in {backoff_time:.1f}s..."
                    )

                time.sleep(backoff_time)

        # If we get here, all retries failed
        raise RateLimitError(
            f"Failed to fetch page {page_num}/{total_pages} after {self.MAX_RETRIES} attempts. "
            f"Last error: {str(last_exception)}"
        ) from last_exception

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


def _get_env_limit() -> int:
    """Get limit from environment variable with validation."""
    try:
        limit_str = os.getenv("LANGSMITH_LIMIT", "0")
        limit = int(limit_str)
        if limit <= 0:
            return 0
        return limit
    except ValueError:
        return 0


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments with environment variable fallbacks.

    Returns:
        Parsed arguments namespace
    """
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Export LangSmith trace data for offline analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage with CLI arguments:
  python export_langsmith_traces.py \\
      --api-key "lsv2_pt_..." \\
      --project "my-project" \\
      --limit 150 \\
      --output "traces_export.json"

Example usage with .env file:
  # Set up .env file with defaults
  echo "LANGSMITH_API_KEY=lsv2_pt_..." >> .env
  echo "LANGSMITH_PROJECT=my-project" >> .env  
  echo "LANGSMITH_LIMIT=150" >> .env
  
  # Then simple usage
  python export_langsmith_traces.py --output traces.json
        """,
    )

    parser.add_argument(
        "--api-key",
        type=str,
        required=False,
        default=os.getenv("LANGSMITH_API_KEY"),
        help="LangSmith API key for authentication (default: LANGSMITH_API_KEY env var)",
    )

    parser.add_argument(
        "--project",
        type=str,
        required=False,
        default=os.getenv("LANGSMITH_PROJECT"),
        help="LangSmith project name or ID (default: LANGSMITH_PROJECT env var)",
    )

    parser.add_argument(
        "--limit",
        type=_positive_int,
        required=False,
        default=_get_env_limit() or None,
        help="Number of most recent traces to export (default: LANGSMITH_LIMIT env var)",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Output JSON file path"
    )

    return parser.parse_args()


def validate_required_args(args: argparse.Namespace) -> None:
    """
    Validate that required arguments are provided via CLI or environment.

    Args:
        args: Parsed command line arguments

    Raises:
        SystemExit: If required arguments are missing
    """
    errors = []

    if not args.api_key:
        errors.append("--api-key is required (or set LANGSMITH_API_KEY in .env)")

    if not args.project:
        errors.append("--project is required (or set LANGSMITH_PROJECT in .env)")

    if not args.limit:
        errors.append("--limit is required (or set LANGSMITH_LIMIT in .env)")

    if errors:
        print("❌ Missing required arguments:", file=sys.stderr)
        for error in errors:
            print(f"   {error}", file=sys.stderr)
        print("\nTip: Create a .env file with your defaults:", file=sys.stderr)
        print("   cp .env.example .env", file=sys.stderr)
        print("   # Edit .env with your values", file=sys.stderr)
        sys.exit(1)


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

        # Validate that required arguments are available
        validate_required_args(args)

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
            # fetch_runs now provides progress updates, so adjust final message
            if len(runs) != args.limit:
                print(f"✓ Fetched {len(runs)} traces (requested {args.limit})")
            else:
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
