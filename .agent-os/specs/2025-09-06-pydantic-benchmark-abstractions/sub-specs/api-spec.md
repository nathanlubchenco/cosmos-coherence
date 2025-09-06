# API Specification

This is the API specification for the spec detailed in @.agent-os/specs/2025-09-06-pydantic-benchmark-abstractions/spec.md

> Created: 2025-09-06
> Version: 1.0.0

## Endpoints

### GET /experiments
Get a list of all benchmark experiments with summary information.

**Response Model:**
```python
class ExperimentSummary(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    status: ExperimentStatus
    total_benchmarks: int
    completion_percentage: float

class ExperimentListResponse(BaseModel):
    experiments: List[ExperimentSummary]
    total_count: int
```

### GET /experiments/{experiment_id}
Get detailed information about a specific experiment including all benchmark results.

**Response Model:**
```python
class ExperimentDetailResponse(BaseModel):
    experiment: ExperimentMetadata
    benchmarks: List[BenchmarkResult]
    summary_statistics: ExperimentSummaryStats
```

### GET /experiments/{experiment_id}/results
Get paginated benchmark results for a specific experiment with optional filtering.

**Query Parameters:**
- `page`: int (default: 1)
- `limit`: int (default: 50, max: 100)
- `benchmark_type`: Optional[str] - filter by benchmark type
- `min_score`: Optional[float] - filter by minimum performance score
- `status`: Optional[str] - filter by result status

**Response Model:**
```python
class BenchmarkResultsResponse(BaseModel):
    results: List[BenchmarkResult]
    pagination: PaginationMetadata
    filters_applied: Dict[str, Any]
```

### GET /experiments/{experiment_id}/export
Export experiment results in various formats for external analysis.

**Query Parameters:**
- `format`: str (json, csv, parquet) - default: json
- `include_metadata`: bool - default: true

**Response:**
- JSON format: `ExperimentExportResponse`
- CSV/Parquet: File download with appropriate headers

### POST /experiments/{experiment_id}/compare
Compare results between multiple experiments or benchmark subsets.

**Request Model:**
```python
class ComparisonRequest(BaseModel):
    comparison_experiments: List[str]
    metrics_to_compare: List[str]
    group_by: Optional[str] = None
```

**Response Model:**
```python
class ComparisonResponse(BaseModel):
    comparison_id: str
    baseline_experiment: str
    compared_experiments: List[str]
    metric_comparisons: List[MetricComparison]
    statistical_significance: Dict[str, float]
```

## Controllers

### ExperimentController
Handles all experiment-related API endpoints.

**Key Methods:**
- `list_experiments()` - Returns paginated list of experiments
- `get_experiment_detail()` - Returns full experiment with results
- `get_experiment_results()` - Returns paginated and filtered results
- `export_experiment()` - Handles various export formats
- `compare_experiments()` - Performs statistical comparison

**Dependencies:**
- `ExperimentRepository` - Data access layer
- `ResultAggregator` - Statistical analysis
- `ExportService` - Format conversion

### ResultController
Specialized controller for result querying and analysis.

**Key Methods:**
- `get_result_by_id()` - Individual result retrieval
- `aggregate_results()` - Real-time aggregation
- `get_performance_trends()` - Time-series analysis

### HealthController
System health and API status endpoints.

**Endpoints:**
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed system status including database connectivity

## Response Models

### Core Response Models
```python
class APIResponse(BaseModel):
    """Base response model for all API endpoints"""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str

class PaginationMetadata(BaseModel):
    page: int
    limit: int
    total_count: int
    total_pages: int
    has_next: bool
    has_previous: bool

class MetricComparison(BaseModel):
    metric_name: str
    baseline_value: float
    comparison_value: float
    percentage_change: float
    absolute_difference: float
    significance_level: Optional[float]
```

### Error Response Models
```python
class APIError(BaseModel):
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ValidationError(APIError):
    field_errors: Dict[str, List[str]]

class NotFoundError(APIError):
    resource_type: str
    resource_id: str
```

## Serialization Format

### JSON Response Format
All API responses follow a consistent structure:

```json
{
    "success": true,
    "timestamp": "2025-09-06T10:30:00Z",
    "request_id": "req_abc123",
    "data": {
        // Actual response data here
    },
    "pagination": {
        // Pagination metadata if applicable
    }
}
```

### Error Response Format
```json
{
    "success": false,
    "timestamp": "2025-09-06T10:30:00Z",
    "request_id": "req_abc123",
    "error": {
        "error_code": "EXPERIMENT_NOT_FOUND",
        "message": "Experiment with ID 'exp_123' was not found",
        "details": {
            "experiment_id": "exp_123"
        }
    }
}
```

### Export Formats

**CSV Export Structure:**
- Flattened benchmark results with metadata columns
- Headers: experiment_id, benchmark_name, benchmark_type, score, duration_ms, memory_usage_mb, timestamp
- One row per benchmark result

**Parquet Export Structure:**
- Nested structure preserving Pydantic model hierarchy
- Optimized for analytical workloads
- Includes schema metadata for type safety

## FastAPI Integration Points

### Middleware Integration
- Request/Response logging middleware
- Performance monitoring middleware
- Error handling middleware with proper Pydantic validation

### Dependency Injection
```python
async def get_experiment_service() -> ExperimentService:
    """Dependency for injecting experiment service"""
    return ExperimentService()

async def get_current_user() -> Optional[User]:
    """Optional authentication dependency"""
    return None  # Authentication not required for benchmark results
```

### Response Serialization
All responses use Pydantic models for automatic:
- JSON serialization
- OpenAPI schema generation
- Input validation
- Type safety

### OpenAPI Documentation
Auto-generated documentation includes:
- Complete schema definitions for all Pydantic models
- Request/response examples
- Parameter descriptions
- Error response schemas
