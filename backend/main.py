"""
MedPilot — FastAPI Backend
Clinical workflow copilot for OpenMRS
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import chat, patients, vitals, conditions, allergies, medications, ingest

app = FastAPI(
    title="MedPilot API",
    description="Clinical workflow copilot powered by OpenAI + OpenMRS",
    version="2.0.0"
)

# Allow Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(chat.router)
app.include_router(patients.router)
app.include_router(vitals.router)
app.include_router(conditions.router)
app.include_router(allergies.router)
app.include_router(medications.router)
app.include_router(ingest.router)


@app.get("/")
def root():
    return {
        "name": "MedPilot API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}
