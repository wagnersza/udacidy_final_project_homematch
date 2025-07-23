"""
HomeMatch - Personalized Real Estate Recommendation System
A RAG-based application that provides personalized real estate recommendations
using Large Language Models and vector databases.
"""

import os
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="HomeMatch",
    description="Personalized Real Estate Recommendation System",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./database/chroma_db")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with user preference form"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "HomeMatch API is running"}


@app.get("/api/listings")
async def get_listings():
    """Get all available listings"""
    # This will be implemented in Phase 2
    return {"message": "Listings endpoint - to be implemented"}


@app.post("/api/search")
async def search_listings(request: Request, preferences: str = Form(...)):
    """Search for listings based on user preferences"""
    # This will be implemented in Phase 3
    return {"message": f"Search functionality - to be implemented. Preferences: {preferences}"}


@app.post("/api/personalize")
async def personalize_listing():
    """Personalize listing descriptions based on user preferences"""
    # This will be implemented in Phase 3
    return {"message": "Personalization endpoint - to be implemented"}


if __name__ == "__main__":
    # Verify environment variables
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_api_key_here":
        print("Warning: OPENAI_API_KEY not set or using placeholder value")
    
    print("Starting HomeMatch application...")
    print(f"OpenAI Base URL: {OPENAI_BASE_URL}")
    print(f"Chroma DB Directory: {CHROMA_PERSIST_DIRECTORY}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
