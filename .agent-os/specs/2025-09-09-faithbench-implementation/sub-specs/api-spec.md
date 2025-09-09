# API Specification

This is the API specification for the spec detailed in @.agent-os/specs/2025-09-09-faithbench-implementation/spec.md

## Endpoints

### POST /api/benchmarks/faithbench/run

**Purpose:** Initiate a FaithBench evaluation run with specified configuration

**Parameters:**
- Request Body (JSON):
  ```json
  {
    "model": "gpt-4",
    "temperature": 0.7,
    "sample_size": 100,
    "categories": ["Science", "History"],  // optional, defaults to all
    "seed": 42,  // optional
    "batch_size": 10,  // optional, defaults to 20
    "checkpoint_enabled": true  // optional, defaults to true
  }
  ```

**Response:**
```json
{
  "run_id": "fb_2025-09-09_12345",
  "status": "running",
  "total_questions": 100,
  "completed": 0,
  "estimated_time_seconds": 300
}
```

**Errors:**
- 400: Invalid configuration parameters
- 409: Another evaluation already running
- 500: Failed to initialize benchmark

### GET /api/benchmarks/faithbench/status/{run_id}

**Purpose:** Check the status of an ongoing FaithBench evaluation

**Parameters:**
- Path: run_id (string)

**Response:**
```json
{
  "run_id": "fb_2025-09-09_12345",
  "status": "running|completed|failed",
  "progress": {
    "total": 100,
    "completed": 45,
    "failed": 2,
    "remaining": 53
  },
  "current_category": "Science",
  "elapsed_seconds": 150,
  "estimated_remaining_seconds": 180
}
```

**Errors:**
- 404: Run ID not found
- 500: Failed to retrieve status

### GET /api/benchmarks/faithbench/results/{run_id}

**Purpose:** Retrieve evaluation results for a completed FaithBench run

**Parameters:**
- Path: run_id (string)
- Query: format (string, optional) - "json" (default) or "csv"

**Response:**
```json
{
  "run_id": "fb_2025-09-09_12345",
  "config": {
    "model": "gpt-4",
    "temperature": 0.7,
    "sample_size": 100
  },
  "metrics": {
    "overall_accuracy": 0.82,
    "hallucination_rate": 0.15,
    "by_category": {
      "Science": {"accuracy": 0.85, "count": 30},
      "History": {"accuracy": 0.78, "count": 25}
    },
    "confidence_distribution": {
      "high": 0.65,
      "medium": 0.25,
      "low": 0.10
    }
  },
  "timestamp": "2025-09-09T12:34:56Z",
  "duration_seconds": 330
}
```

**Errors:**
- 404: Run ID not found or results not available
- 400: Invalid format parameter
- 500: Failed to retrieve results

### DELETE /api/benchmarks/faithbench/cancel/{run_id}

**Purpose:** Cancel an ongoing FaithBench evaluation

**Parameters:**
- Path: run_id (string)

**Response:**
```json
{
  "run_id": "fb_2025-09-09_12345",
  "status": "cancelled",
  "questions_completed": 45,
  "partial_results_saved": true
}
```

**Errors:**
- 404: Run ID not found
- 409: Evaluation already completed or cancelled
- 500: Failed to cancel evaluation

### GET /api/benchmarks/faithbench/dataset/info

**Purpose:** Get information about the FaithBench dataset

**Parameters:** None

**Response:**
```json
{
  "total_questions": 1000,
  "categories": {
    "Science": 200,
    "History": 150,
    "Geography": 150,
    "Current_Events": 200,
    "Other": 300
  },
  "difficulty_levels": ["easy", "medium", "hard"],
  "dataset_version": "1.0",
  "last_updated": "2025-01-15"
}
```

**Errors:**
- 503: Dataset not available
- 500: Failed to load dataset information

## Integration with CLI

The API endpoints are called internally by the CLI commands:
- `cosmos-coherence benchmark run faithbench` → POST /api/benchmarks/faithbench/run
- `cosmos-coherence benchmark status <run_id>` → GET /api/benchmarks/faithbench/status/{run_id}
- `cosmos-coherence benchmark results <run_id>` → GET /api/benchmarks/faithbench/results/{run_id}

## Rate Limiting

- Maximum 5 concurrent evaluation runs per instance
- Request rate limit: 100 requests per minute per client
- Checkpoint saves throttled to once per minute

## Authentication

Currently uses existing authentication mechanism (if implemented). For initial implementation, no additional authentication required beyond existing system.
