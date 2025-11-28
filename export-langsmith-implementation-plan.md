# PDCA PLAN Phase - LangSmith Data Export Script

**Date:** 2025-11-28
**Framework:** PDCA (Plan-Do-Check-Act)
**Phase:** PLAN - Analysis & Implementation Plan
**Estimated Duration:** 2-3 hours (implementation)

---

## 1. ANALYSIS

### 1.1 Objectives

Create a Python CLI script that:
1. Exports LangSmith trace data using the Python SDK
2. Retrieves N most recent traces from a specified project
3. Exports to structured JSON format with metadata
4. Implements robust error handling and rate limiting
5. Provides clear progress feedback to users

### 1.2 Requirements Mapping

**Functional Requirements:**
- **FR1**: Initialize LangSmith client with API key and endpoint
- **FR2**: Query traces using `client.list_runs()` with filters
- **FR3**: Handle rate limiting with exponential backoff
- **FR4**: Export to JSON with specified schema structure
- **FR5**: Comprehensive error handling (auth, network, project not found)
- **FR6**: Data validation (verify traces retrieved, validate JSON)

**Non-Functional Requirements:**
- **NFR1**: Performance (< 5 min for 200 traces, progress indicator)
- **NFR2**: Usability (clear output, easy parameter modification)
- **NFR3**: Maintainability (type hints, comments, modular structure)

### 1.3 Technical Constraints

- **LangSmith Plan**: Individual Developer (no bulk export API)
- **Export Method**: SDK-based using `langsmith` Python package
- **API Endpoint**: `https://api.smith.langchain.com` (SDK default)
- **Rate Limiting**: Must respect API limits with throttling/backoff

### 1.4 Key Technical Challenges

1. **Rate Limiting**: Implement exponential backoff for API throttling
2. **Nested Relationships**: Handle parent/child run relationships
3. **Progress Indication**: User feedback for long-running exports
4. **Error Recovery**: Continue export if individual trace fails
5. **Data Validation**: Ensure complete and valid JSON output

---

## 2. ARCHITECTURE APPROACH

### 2.1 Modular Structure

```
export_langsmith_traces.py (main script)
├── parse_arguments()           # CLI argument parsing
├── LangSmithExporter class
│   ├── __init__()             # Client initialization
│   ├── fetch_runs()           # Query with rate limiting
│   ├── format_trace_data()    # Transform to output schema
│   ├── export_to_json()       # Save formatted data
│   └── _handle_rate_limit()   # Exponential backoff helper
└── main()                      # Orchestration and entry point

test_export_langsmith_traces.py (unit tests)
├── TestArgumentParsing
├── TestLangSmithExporter
│   ├── test_client_init
│   ├── test_fetch_runs
│   ├── test_rate_limiting
│   ├── test_format_trace_data
│   └── test_export_to_json
└── TestErrorHandling
```

### 2.2 Dependencies

**Required:**
- `langsmith` - LangSmith Python SDK for API interaction
- `argparse` - CLI argument parsing (stdlib)
- `json` - JSON serialization (stdlib)
- `datetime` - Timestamps (stdlib)
- `time` - Rate limit backoff delays (stdlib)

**Optional:**
- `tqdm` - Progress bar for exports > 50 traces
- `python-dotenv` - Environment variable management

### 2.3 Output JSON Schema

```json
{
  "export_metadata": {
    "export_timestamp": "ISO-8601 datetime",
    "project_name": "string",
    "total_traces": "integer",
    "langsmith_api_version": "string"
  },
  "traces": [
    {
      "id": "string",
      "name": "string",
      "start_time": "ISO-8601 datetime",
      "end_time": "ISO-8601 datetime",
      "duration_seconds": "float",
      "status": "success|error",
      "inputs": "object",
      "outputs": "object",
      "error": "string|null",
      "run_type": "string",
      "child_runs": "array"
    }
  ]
}
```

### 2.4 Testing Strategy (TDD)

**Approach:** Strict Test-Driven Development
- One failing test at a time
- Red → Green → Refactor cycle
- Mock LangSmith client for unit tests
- Test error scenarios comprehensively

**Test Coverage:**
- Unit tests for each method
- Error handling tests (auth, rate limit, network)
- Data transformation tests
- Edge cases (zero traces, missing fields)
- Integration test (optional, requires API key)

---

## 3. IMPLEMENTATION PLAN

**Working Agreement:** Strict TDD - one failing test at a time, no exceptions.

### Step 1: Project Setup & Dependencies
**Duration:** 10 minutes

**Tasks:**
- [ ] Set up virtual environment using `uv` or `venv`
  - Option A: `uv venv` (faster, modern)
  - Option B: `python -m venv .venv` (standard library)
- [ ] Activate virtual environment
- [ ] Create `requirements.txt` with pinned versions
  - langsmith
  - tqdm (optional)
  - python-dotenv (optional)
- [ ] Install dependencies: `uv pip install -r requirements.txt` or `pip install -r requirements.txt`
- [ ] Create `export_langsmith_traces.py` skeleton with imports
- [ ] Create `test_export_langsmith_traces.py` with test framework
- [ ] Create `.env.example` for API key template
- [ ] Update `.gitignore` to exclude `.venv/` and `.env`

**Checkpoint:** ✓ Virtual environment created, dependencies installed, files created

---

### Step 2: CLI Argument Parsing (TDD)
**Duration:** 15 minutes

**TDD Cycle:**
1. [ ] **RED**: Write test for `parse_arguments()` with required args
   - Test: `--api-key`, `--project`, `--limit`, `--output`
   - Expected: Returns namespace with all arguments
2. [ ] **GREEN**: Implement `parse_arguments()` using `argparse`
3. [ ] **RED**: Write test for missing required argument
   - Expected: Raises SystemExit with error message
4. [ ] **GREEN**: Add required=True to arguments
5. [ ] **REFACTOR**: Add argument validation (limit > 0)

**Checkpoint:** ✓ CLI arguments parseable with validation

---

### Step 3: LangSmith Client Initialization (TDD)
**Duration:** 20 minutes

**TDD Cycle:**
1. [ ] **RED**: Write test for `LangSmithExporter.__init__()`
   - Test: Initializes with valid API key
   - Expected: Client object created
2. [ ] **GREEN**: Implement class with client initialization
3. [ ] **RED**: Write test for invalid API key (auth error)
   - Expected: Raises custom AuthenticationError
4. [ ] **GREEN**: Add try/catch for auth errors in `__init__`
5. [ ] **REFACTOR**: Extract API endpoint as class constant

**Checkpoint:** ✓ Client initializes correctly with error handling

---

### Step 4: Fetch Runs with Rate Limiting (TDD)
**Duration:** 30 minutes

**TDD Cycle:**
1. [ ] **RED**: Write test for `fetch_runs(project, limit)`
   - Mock: `client.list_runs()` returns list of Run objects
   - Expected: Returns list of runs
2. [ ] **GREEN**: Implement basic `fetch_runs()` method
3. [ ] **RED**: Write test for rate limit error (429 response)
   - Mock: Raises rate limit exception
   - Expected: Retries with exponential backoff
4. [ ] **GREEN**: Implement `_handle_rate_limit()` with retry logic
5. [ ] **RED**: Write test for max retries exceeded
   - Expected: Raises RateLimitError after max attempts
6. [ ] **GREEN**: Add max retry limit (e.g., 5 attempts)
7. [ ] **REFACTOR**: Extract backoff parameters as class constants

**Checkpoint:** ✓ Runs fetched with automatic rate limit handling

---

### Step 5: Data Formatting (TDD)
**Duration:** 25 minutes

**TDD Cycle:**
1. [ ] **RED**: Write test for `format_trace_data(runs)`
   - Mock: List of Run objects with all fields
   - Expected: Returns dict matching output schema
2. [ ] **GREEN**: Implement transformation logic
3. [ ] **RED**: Write test for missing/null fields
   - Mock: Run with null outputs, no error field
   - Expected: Uses null/default values, doesn't crash
4. [ ] **GREEN**: Add safe field extraction with `.get()` or `getattr()`
5. [ ] **RED**: Write test for child_runs relationship
   - Mock: Parent run with nested child runs
   - Expected: Includes child_runs array
6. [ ] **GREEN**: Handle nested run extraction
7. [ ] **REFACTOR**: Extract field mapping to helper method

**Checkpoint:** ✓ Data formatted per requirements (FR4)

---

### Step 6: JSON Export (TDD)
**Duration:** 20 minutes

**TDD Cycle:**
1. [ ] **RED**: Write test for `export_to_json(data, filepath)`
   - Test: Creates JSON file with correct structure
   - Expected: File exists with valid JSON
2. [ ] **GREEN**: Implement JSON file writing with metadata
3. [ ] **RED**: Write test for file write permission error
   - Mock: Raises PermissionError
   - Expected: Raises custom ExportError with helpful message
4. [ ] **GREEN**: Add try/catch for file I/O errors
5. [ ] **REFACTOR**: Add pretty-printing (indent=2)

**Checkpoint:** ✓ JSON export working with error handling

---

### Step 7: Progress Indication (NFR1)
**Duration:** 15 minutes

**TDD Cycle:**
1. [ ] **RED**: Write test for progress callback in `fetch_runs()`
   - Test: Calls progress callback for each batch
   - Expected: Progress function invoked N times
2. [ ] **GREEN**: Add optional progress callback parameter
3. [ ] **GREEN**: Implement `tqdm` progress bar wrapper
4. [ ] **REFACTOR**: Make progress bar optional (only if tqdm installed)

**Checkpoint:** ✓ Progress feedback implemented

---

### Step 8: Error Scenarios (FR5)
**Duration:** 25 minutes

**TDD Cycle:**
1. [ ] **RED**: Write test for project not found
   - Mock: API returns 404
   - Expected: Raises ProjectNotFoundError
2. [ ] **GREEN**: Handle project not found exception
3. [ ] **RED**: Write test for network timeout
   - Mock: Raises timeout exception
   - Expected: Retries, then raises NetworkError
4. [ ] **GREEN**: Add timeout handling with retry
5. [ ] **RED**: Write test for zero traces returned
   - Expected: Logs warning, creates valid JSON with empty array
6. [ ] **GREEN**: Add validation for empty results
7. [ ] **REFACTOR**: Consolidate error handling

**Checkpoint:** ✓ All error scenarios handled gracefully

---

### Step 9: Main Orchestration
**Duration:** 20 minutes

**TDD Cycle:**
1. [ ] **RED**: Write integration test for `main()` function
   - Mock: All components working together
   - Expected: Completes without error
2. [ ] **GREEN**: Implement `main()` orchestrating all steps
3. [ ] **GREEN**: Add logging for debugging (optional)
4. [ ] **REFACTOR**: Extract magic numbers to constants

**Checkpoint:** ✓ Complete script working end-to-end

---

### Step 10: Documentation & Usability (NFR2, NFR3)
**Duration:** 15 minutes

**Tasks:**
- [ ] Add docstrings to all functions and class methods
- [ ] Add module-level usage instructions in script header
- [ ] Update README.md with:
  - Setup instructions
  - Usage examples
  - Environment variable configuration
  - Troubleshooting guide
- [ ] Add example `.env` file for API key storage
- [ ] Add inline comments for complex logic

**Checkpoint:** ✓ Script fully documented and user-friendly

---

## 4. DEFINITION OF DONE

### Quality Gates

**Functional Completeness:**
- [ ] All functional requirements (FR1-FR6) implemented
- [ ] All non-functional requirements (NFR1-NFR3) met
- [ ] All unit tests passing (>80% coverage)
- [ ] Integration test successful with actual API (manual)

**Code Quality:**
- [ ] Type hints on all function signatures
- [ ] Docstrings on all public functions/methods
- [ ] No pylint/flake8 warnings
- [ ] Error handling for all identified failure modes

**User Experience:**
- [ ] Progress indication for exports > 50 traces
- [ ] Clear error messages for common failures
- [ ] README with setup and usage instructions
- [ ] Example output JSON file included

**Testing:**
- [ ] Unit tests for each component
- [ ] Error scenario tests passing
- [ ] Edge case tests (empty results, missing fields)
- [ ] Manual test with real LangSmith API successful

### PDCA Working Agreements

**Commitments:**
1. **Strict TDD**: One failing test at a time, no exceptions
2. **Respect Modularity**: Separate concerns (fetch, format, export)
3. **Intervene on Sprawl**: Keep functions focused and single-purpose
4. **No Premature Optimization**: Get it working correctly first
5. **Active Oversight**: Review each implementation step
6. **Immediate Intervention**: Stop and correct on process violations

---

## 5. RISK MITIGATION

### Identified Risks

1. **Risk**: API rate limiting blocks export
   - **Mitigation**: Exponential backoff with max retries (Step 4)

2. **Risk**: Large exports consume excessive memory
   - **Mitigation**: Consider pagination/streaming in future iteration

3. **Risk**: API schema changes break parsing
   - **Mitigation**: Safe field extraction with defaults (Step 5)

4. **Risk**: Network failures mid-export
   - **Mitigation**: Retry logic with timeouts (Step 8)

---

## 6. NEXT STEPS

After plan approval:
1. **Proceed to DO phase** starting with Step 1
2. **Follow TDD strictly**: Red → Green → Refactor
3. **Use checklist**: Mark off each sub-task as completed
4. **Checkpoint validation**: Verify each checkpoint before proceeding
5. **Context drift recovery**: If scope expands, return to PLAN phase

**Ready to begin implementation?** Confirm plan approval to proceed to DO phase.
