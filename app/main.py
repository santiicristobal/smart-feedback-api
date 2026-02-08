from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.model import SentimentModel
from app.preprocessing import clean_text


app = FastAPI(
    title="Smart Feedback API",
    description="API para análisis de sentimiento de feedback de usuarios",
    version="1.0.0"
)


sentiment_model = SentimentModel()


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    sentiment: str
    score: float


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_sentiment(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    clean = clean_text(request.text)
    result = sentiment_model.predict(clean)

    return result
