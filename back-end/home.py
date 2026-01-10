from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from datetime import datetime
import hashlib
import uvicorn
from openai import OpenAI
from typing import Dict

app = FastAPI()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str
    title: str


def analyze_bias(article_dict: dict) -> dict:
    text_to_analyze = article_dict["text"]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that analyzes news articles for bias. "
                    "Highlight loaded language, bias, and missing perspectives. "
                    "Return a JSON object with keys: bias_score, highlights, explanation, "
                    "missing_perspectives, sources (list of URLs supporting claims)."
                )
            },
            {
                "role": "user",
                "content": f"Analyze this news article:\n\n{text_to_analyze}"
            }
        ],
        response_format="json"
    )

    ai_json = response.output[0].content[0].text
    if isinstance(ai_json, str):
        try:
            ai_json = json.loads(ai_json)
        except json.JSONDecodeError:
            ai_json = {"error": "AI did not return valid JSON", "raw_text": ai_json}
    return ai_json



@app.post("/receive_json")
async def receive_json(article: AnalyzeRequest):
    # Prepare dict to save
    article_dict = article.model_dump()

    ai_response = analyze_bias(article_dict)

    return {
        "status": "analyzed",
        "article": article_dict,
        "ai_result": ai_response
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True)


