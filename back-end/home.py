from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import uvicorn
from fastapi import APIRouter, Depends
from typing import list, dict
from textblob import TextBlob
import httpx





if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True)

