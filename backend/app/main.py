"""
FastAPI Application Entrypoint — Arabic Handwritten Letter AI Service
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.recognition import router as recognition_router

app = FastAPI(
    title="Arabic Handwritten Letter Recognition AI Service",
    description="Independent AI module for recognizing 28 isolated Arabic handwritten characters from camera captures.",
    version="0.1.0",
)

# CORS Middleware (Enable cross-origin requests from web demos, mobile and tablet browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(recognition_router)

# Mount demo folder for convenient browser demonstration
DEMO_DIR = Path(__file__).resolve().parent.parent.parent / "demo"
if DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(DEMO_DIR), html=True), name="demo")


@app.get("/")
def root():
    return {
        "service": "Arabic Handwritten Letter AI",
        "status": "online",
        "docs_url": "/docs",
        "demo_url": "/demo/index.html",
    }
