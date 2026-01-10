from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import uvicorn
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import hashlib






if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True)

