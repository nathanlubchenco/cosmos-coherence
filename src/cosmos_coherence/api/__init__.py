"""FastAPI application for Cosmos Coherence."""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Cosmos Coherence API",
    description="LLM Hallucination Detection Framework API",
    version="0.1.0",
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Cosmos Coherence API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(status_code=200, content={"status": "healthy", "service": "api"})
