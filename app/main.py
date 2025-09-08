# app/main.py
from typing import Optional, Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .verify import ImageMeta, VerifyConfig, verify_pair  # <- wire-in

app = FastAPI(title="Civic Issue Before/After Verifier", version="0.1.0")


# ---------- Schemas ----------
class ImageIn(BaseModel):
    lat: Optional[float] = Field(default=None)
    lng: Optional[float] = Field(default=None)
    timestamp: Optional[str] = Field(
        default=None, description="ISO 8601, e.g. 2025-09-01T10:20:00Z"
    )
    image_url: str
    description: Optional[str] = None


class VerifyRequest(BaseModel):
    before: ImageIn
    after: ImageIn


# ---------- Routes ----------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "civic-verifier", "version": "0.1.0"}


@app.post("/ml/verify-change")
def verify_change(payload: VerifyRequest) -> Dict[str, Any]:
    """
    Verifies: same location? and issue resolved?
    """
    try:
        before = ImageMeta(
            lat=payload.before.lat,
            lng=payload.before.lng,
            timestamp=payload.before.timestamp,
            image_url=payload.before.image_url,
            description=payload.before.description,
        )
        after = ImageMeta(
            lat=payload.after.lat,
            lng=payload.after.lng,
            timestamp=payload.after.timestamp,
            image_url=payload.after.image_url,
            description=payload.after.description,
        )

        cfg = VerifyConfig()  # tweak thresholds here later if needed
        result = verify_pair(before, after, cfg)
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


