# LangSmith Data Export Script - Requirements

## Purpose
Export workflow trace data from LangSmith projects for offline analysis.

## Constraints
- **LangSmith Plan:** Individual Developer (no bulk export feature available)
- **Export Method:** SDK-based using `list_runs()` from LangSmith Python SDK
- **API Endpoint:** `https://api.smith.langchain.com` (default for SDK)
- **Rate Limiting:** Must respect LangSmith API rate limits (throttle requests if needed)

## Required Inputs
1. **LangSmith API Key:** Authentication credential
2. **Project Name/ID:** LangSmith project identifier
3. **Number of Traces:** Count of most recent traces to export (N parameter)
4. **Optional Filters:** Date range, status filters (if needed)

## Functional Requirements

### FR1: Initialize LangSmith Client
- Import `langsmith` Python SDK
- Initialize client with API key
- Set API URL: `https://api.smith.langchain.com` (default)

### FR2: Query Traces
- Use `client.list_runs()` method
- Filter by project name/ID
- Retrieve most recent N traces (ordered by start time descending)
- Include all run metadata, timing, costs, errors

### FR3: Handle Rate Limiting
- Implement exponential backoff for rate limit errors
- Optionally use `ThreadPoolExecutor` with `max_workers=2` (based on LangSmith docs)
- Log progress for long-running exports

### FR4: Export to JSON
- Convert trace data to JSON format
- Save to file: `langsmith_traces_{project_name}_{timestamp}.json`
- Include:
  - Run ID, start/end timestamps, duration
  - Run status (success, error)
  - Input/output data (if available)
  - Error messages (if present)
  - Token usage and costs (if tracked)
  - Parent/child run relationships (for nested nodes)

### FR5: Error Handling
- Handle authentication errors (invalid API key)
- Handle project not found errors
- Handle network timeouts
- Log errors to console with clear messages
- Don't fail entire export if individual trace retrieval fails

### FR6: Data Validation
- Verify at least 1 trace was retrieved
- Validate JSON structure before saving
- Report total traces exported vs requested

## Output Format

### JSON Structure
```json
{
  "export_metadata": {
    "export_timestamp": "2024-12-02T12:00:00Z",
    "project_name": "project_name",
    "total_traces": 150,
    "langsmith_api_version": "0.1.x"
  },
  "traces": [
    {
      "id": "run_id",
      "name": "workflow_name",
      "start_time": "2024-11-28T10:00:00Z",
      "end_time": "2024-11-28T10:15:00Z",
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

## Non-Functional Requirements

### NFR1: Performance
- Export should complete within reasonable time (< 5 minutes for 200 traces)
- Display progress indicator for exports > 50 traces

### NFR2: Usability
- Clear console output showing progress
- Easy to modify parameters (API key, project name, count)
- Include usage instructions in script header

### NFR3: Maintainability
- Well-commented code
- Modular structure (separate functions for query, format, save)
- Use type hints

## Dependencies
- `langsmith` Python SDK (pip install langsmith)
- `json` (standard library)
- `datetime` (standard library)
- Optional: `tqdm` for progress bar

## Usage Example
```python
python export_langsmith_traces.py \
  --api-key "lsv2_pt_..." \
  --project "your-project-name" \
  --limit 150 \
  --output "traces_export.json"
```

## API Reference
- **LangSmith API Docs:** https://docs.smith.langchain.com/
- **SDK Method:** `client.list_runs(project_name="...", limit=N)`
- **API Endpoint:** https://api.smith.langchain.com/api/v1/runs/query (used internally by SDK)

## Success Criteria
- Script successfully authenticates with LangSmith API
- Retrieves requested number of traces (or all available if fewer)
- Exports to valid JSON file
- Handles errors gracefully without crashing
- Provides clear progress feedback
- Output file size is reasonable and readable
