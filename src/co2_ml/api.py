"""HTTP service.  Run:  uvicorn co2_ml.api:app --port 8000   then open http://localhost:8000/docs"""
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .service import Service

app = FastAPI(title="CO2 Emissions Model + RAG assistant", version="1.0.0")
svc = Service()

class Vehicle(BaseModel):
    vehicle_class: str = Field(examples=["SUV - SMALL"])
    engine_size_l: float = Field(gt=0, lt=12, examples=[2.0])
    cylinders: int = Field(ge=2, le=16, examples=[4])
    transmission: str = Field(examples=["AS6"])
    fuel_type: str = Field(examples=["X"])
    fuel_consumption_comb: Optional[float] = Field(None, gt=0, lt=40, description="L/100km; omit to use the spec-only model")

class Question(BaseModel):
    question: str = Field(min_length=3)
    vehicle: Optional[Vehicle] = None

@app.get("/health")
def health():
    return {"status": "ok", "champions": {s: svc.reg.champion_version(s) for s in ("spec_only", "spec_plus_consumption")}}

@app.post("/predict")
def predict(v: Vehicle, explain: bool = False):
    try: return svc.predict(v.model_dump(), explain)
    except FileNotFoundError as e: raise HTTPException(503, str(e))

@app.post("/ask")
def ask(q: Question):
    try: return svc.ask(q.question, q.vehicle.model_dump() if q.vehicle else None)
    except FileNotFoundError as e: raise HTTPException(503, str(e))

@app.post("/reindex")
def reindex():
    svc.rag.reindex(); return {"chunks": len(svc.rag.index.chunks)}
