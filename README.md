# LangSmith Data Export Script

Export workflow trace data from LangSmith projects for offline analysis and review.

## Overview

This Python script exports trace data from LangSmith using the SDK API, designed for users on Individual Developer plans without bulk export features. It provides robust error handling, rate limiting, and progress feedback for reliable data exports.

## Features

- Export N most recent traces from any LangSmith project
- Automatic rate limiting with exponential backoff
- Progress indication for long-running exports
- Comprehensive error handling (auth, network, rate limits)
- Structured JSON output with metadata
- Type-safe implementation with full type hints
- Test-driven development with pytest suite

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
source .venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # Linux/Mac
```

**Option B: Using venv**
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
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

### Basic Usage

```bash
python export_langsmith_traces.py \
  --api-key "lsv2_pt_..." \
  --project "your-project-name" \
  --limit 150 \
  --output "traces_export.json"
```

### Parameters

- `--api-key` (required): LangSmith API key for authentication
- `--project` (required): LangSmith project name or ID
- `--limit` (required): Number of most recent traces to export (must be > 0)
- `--output` (required): Output JSON file path

### Example

```bash
python export_langsmith_traces.py \
  --api-key "lsv2_pt_abc123..." \
  --project "neota-aesp-project" \
  --limit 200 \
  --output "neota_traces_2025-11-28.json"
```

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

### Completed
- ✅ Project setup with virtual environment
- ✅ Dependencies configuration
- ✅ CLI argument parsing with validation
- ✅ Test suite foundation
- ✅ Development documentation

### In Progress
- 🔄 LangSmith client initialization
- 🔄 Run fetching with rate limiting
- 🔄 Data formatting and transformation
- 🔄 JSON export functionality

### Planned
- ⏳ Progress indication
- ⏳ Error scenario handling
- ⏳ Main orchestration
- ⏳ End-to-end integration

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

[Specify your license here]

## Contributing

This project was developed using Claude Code with the PDCA framework. See PLAN.md for implementation details.

## References

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangSmith Python SDK](https://github.com/langchain-ai/langsmith-sdk)
- [PDCA Framework](https://github.com/kenjudy/human-ai-collaboration-process)
