# LangSmith Data Export Script

Export workflow trace data from LangSmith projects for offline analysis and review.

## Overview

This Python script exports trace data from LangSmith using the SDK API, designed for users on Individual Developer plans without bulk export features. It provides robust error handling, rate limiting, and progress feedback for reliable data exports.

## Features

- Export N most recent traces from any LangSmith project
- **Automatic pagination** - Handles large exports (> 100 records) seamlessly with progress indication
- **Environment variable support** - Configure once via `.env` file for simplified usage
- Automatic rate limiting with exponential backoff
- Progress indication for long-running exports
- Comprehensive error handling (auth, network, rate limits)
- Structured JSON output with metadata
- Type-safe implementation with full type hints
- Test-driven development with pytest suite (33 tests, high coverage)

## Requirements

- Python 3.8 or higher
- LangSmith API key (Individual Developer plan or higher)
- Virtual environment manager (uv or venv)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd export-langsmith-data
```

### 2. Set up virtual environment

**Option A: Using uv (recommended)**
```bash
uv venv
.venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # Linux/Mac
```

**Option B: Using venv**
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # Linux/Mac
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
# or
pip install -r requirements.txt
```

### 4. Configure API key

Create a `.env` file from the template:
```bash
cp .env.example .env
```

Edit `.env` and add your LangSmith API key:
```env
LANGSMITH_API_KEY=lsv2_pt_your_api_key_here
```

Get your API key from: https://smith.langchain.com/settings

## Usage

### Option 1: Using Environment Variables (Recommended)

Set up your `.env` file once:

```bash
cp .env.example .env
# Edit .env with your values:
# LANGSMITH_API_KEY=lsv2_pt_your_key_here
# LANGSMITH_PROJECT=your-project-name
# LANGSMITH_LIMIT=150
```

Then use simple commands:

```bash
python export_langsmith_traces.py --output "traces_export.json"
```

### Option 2: Using Command Line Arguments

```bash
python export_langsmith_traces.py \
  --api-key "lsv2_pt_..." \
  --project "your-project-name" \
  --limit 150 \
  --output "traces_export.json"
```

### Option 3: Mixed Usage (Override Environment Variables)

```bash
# Override just the project while using env vars for api-key and limit
python export_langsmith_traces.py \
  --project "different-project" \
  --output "traces_export.json"
```

### Parameters

- `--api-key` (optional): LangSmith API key for authentication (default: `LANGSMITH_API_KEY` env var)
- `--project` (optional): LangSmith project name or ID (default: `LANGSMITH_PROJECT` env var)
- `--limit` (optional): Number of most recent traces to export (default: `LANGSMITH_LIMIT` env var)
  - For limits > 100, the tool automatically handles pagination across multiple API calls
  - If fewer records exist in the project, you'll receive a warning and all available records
- `--output` (required): Output JSON file path

**Note**: While the CLI arguments are now optional, the values must be provided either via command line or environment variables.

### Examples

**Using environment variables:**
```bash
# Set up .env file once
echo "LANGSMITH_API_KEY=lsv2_pt_abc123..." >> .env
echo "LANGSMITH_PROJECT=neota-aesp-project" >> .env
echo "LANGSMITH_LIMIT=200" >> .env

# Simple usage
python export_langsmith_traces.py --output "neota_traces_2025-11-28.json"
```

**Using CLI arguments:**
```bash
python export_langsmith_traces.py \
  --api-key "lsv2_pt_abc123..." \
  --project "neota-aesp-project" \
  --limit 200 \
  --output "neota_traces_2025-11-28.json"
```

**Mixed usage:**
```bash
# Use env vars for api-key and project, override limit
python export_langsmith_traces.py \
  --limit 500 \
  --output "large_export.json"
```

## Pagination for Large Exports

The LangSmith API limits results to 100 records per call. This tool automatically handles pagination for larger exports with progress indication:

**Example: Exporting 500 records**
```bash
python export_langsmith_traces.py --limit 500 --output large_export.json
```

**Output:**
```
🚀 Exporting 500 traces from project 'my-project'...
✓ Connected to LangSmith API
📥 Fetching traces...
  📄 Fetching 500 runs across 5 pages...
    ✓ Page 1/5: 100 runs (Total: 100)
    ✓ Page 2/5: 100 runs (Total: 200)
    ✓ Page 3/5: 100 runs (Total: 300)
    ✓ Page 4/5: 100 runs (Total: 400)
    ✓ Page 5/5: 100 runs (Total: 500)
✓ Fetched 500 traces
🔄 Formatting trace data...
✓ Data formatted
💾 Exporting to large_export.json...
✅ Export complete! Saved to large_export.json
```

**If Project Has Fewer Records:**
```
⚠️  Warning: Fetched 250 runs (requested 500)
```

**Pagination Features:**
- Automatic chunking into 100-record pages
- Progress indication for multi-page exports
- Rate limiting between pages (500ms delay)
- Retry logic per page for reliability
- Warning when fewer records available than requested

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "export_metadata": {
    "export_timestamp": "2025-11-28T12:00:00Z",
    "project_name": "project-name",
    "total_traces": 150,
    "langsmith_api_version": "0.4.x"
  },
  "traces": [
    {
      "id": "run_id",
      "name": "workflow_name",
      "start_time": "2025-11-28T10:00:00Z",
      "end_time": "2025-11-28T10:15:00Z",
      "duration_seconds": 900,
      "status": "success",
      "inputs": {},
      "outputs": {},
      "error": null,
      "run_type": "chain",
      "child_runs": []
    }
  ]
}
```

## Development

### Running Tests

```bash
# Run all tests
pytest test_export_langsmith_traces.py -v

# Run specific test class
pytest test_export_langsmith_traces.py::TestArgumentParsing -v

# Run with coverage
pytest --cov=export_langsmith_traces test_export_langsmith_traces.py
```

### Project Structure

```
export-langsmith-data/
├── .env.example              # API key configuration template
├── .gitignore               # Git ignore patterns
├── requirements.txt         # Python dependencies
├── PLAN.md                  # PDCA implementation plan
├── export-langsmith-requirements.md  # Requirements specification
├── export_langsmith_traces.py        # Main script
├── test_export_langsmith_traces.py   # Test suite
└── README.md                # This file
```

### Development Approach

This project follows the **PDCA (Plan-Do-Check-Act) framework** with strict Test-Driven Development:
- **Plan**: Comprehensive analysis and implementation plan (see PLAN.md)
- **Do**: TDD implementation with Red-Green-Refactor cycles
- **Check**: Validation against requirements and quality standards
- **Act**: Retrospection and continuous improvement

## Implementation Status

### ✅ Complete - Production Ready

All core features implemented and tested:

- ✅ Project setup with virtual environment (uv/venv)
- ✅ Dependencies configuration with CI/CD quality gates
- ✅ CLI argument parsing with validation
- ✅ **Environment variable support** - Optional `.env` file configuration for simplified usage
- ✅ **Automatic pagination** - Handles API 100-record limit with multi-page fetching and progress indication
- ✅ LangSmith client initialization with authentication
- ✅ Run fetching with exponential backoff rate limiting
- ✅ Data formatting and transformation with safe field extraction
- ✅ JSON export functionality with error handling
- ✅ Comprehensive error scenario handling
- ✅ Main orchestration with user-friendly progress feedback
- ✅ End-to-end integration testing
- ✅ Test suite: 33 tests (25 original + 8 pagination tests), high coverage
- ✅ Code quality: Black, Ruff, mypy, Bandit, Safety checks passing

### Optional Features Not Implemented

- ⏸️ Progress indication (tqdm) - Skipped in favor of simple console output

## Troubleshooting

### Authentication Errors
- Verify your API key is correct in `.env` or command line
- Check API key has not expired at https://smith.langchain.com/settings

### Rate Limit Errors
- Script automatically retries with exponential backoff
- Consider reducing export frequency if hitting limits repeatedly

### Network Errors
- Check internet connectivity
- Verify access to https://api.smith.langchain.com
- Check firewall/proxy settings

## License

Property of Stride http://www.stride.build

## Contributing

This project was developed using Claude Code with the PDCA framework. See export-langsmith-implementation-plan.md for implementation details.

## References

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangSmith Python SDK](https://github.com/langchain-ai/langsmith-sdk)
- [PDCA Framework](https://github.com/kenjudy/human-ai-collaboration-process)
