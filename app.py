"""
HomeMatch - Personalized Real Estate Recommendation System
A RAG-based application that provides personalized real estate recommendations
using Large Language Models and vector databases.
"""

import os
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import uvicorn

# Import our modules
from modules.data_generator import DataGenerator
from modules.vector_store import VectorStore

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="HomeMatch",
    description="Personalized Real Estate Recommendation System",
    version="1.0.0"
)

# Initialize modules
data_generator = DataGenerator()
vector_store = VectorStore()

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
    try:
        # First, try to load from file
        listings = data_generator.load_listings_from_file()
        
        if not listings:
            # If no file exists, check vector database
            db_listings = vector_store.get_all_listings()
            if db_listings:
                # Convert vector store format to listing format
                listings = []
                for db_listing in db_listings:
                    listing = {
                        "id": db_listing["id"],
                        **db_listing["metadata"]
                    }
                    listings.append(listing)
        
        return JSONResponse({
            "count": len(listings),
            "listings": listings
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving listings: {str(e)}")


@app.post("/api/generate-listings")
async def generate_listings(count: int = 12):
    """Generate new synthetic real estate listings"""
    try:
        print(f"🏠 Generating {count} new listings...")
        
        # Generate listings
        listings = await data_generator.generate_multiple_listings(count)
        
        # Save to file
        filepath = data_generator.save_listings_to_file(listings)
        
        # Add to vector database
        added_count = vector_store.add_multiple_listings(listings)
        
        return JSONResponse({
            "message": f"Successfully generated {len(listings)} listings",
            "count": len(listings),
            "saved_to_file": filepath,
            "added_to_vector_db": added_count,
            "listings": listings[:3]  # Return first 3 as sample
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating listings: {str(e)}")


@app.post("/api/search")
async def search_listings(request: Request, preferences: str = Form(...)):
    """Search for listings based on user preferences"""
    try:
        if not preferences.strip():
            raise HTTPException(status_code=400, detail="Preferences cannot be empty")
        
        # Perform semantic search
        results = vector_store.search_listings(preferences, n_results=5)
        
        if not results:
            # If no results from vector search, return all listings as fallback
            all_listings = data_generator.load_listings_from_file()
            return JSONResponse({
                "message": "No semantic matches found. Showing all available listings.",
                "preferences": preferences,
                "count": len(all_listings),
                "results": all_listings[:5]  # Return first 5
            })
        
        # Format results for frontend
        formatted_results = []
        for result in results:
            formatted_result = {
                "id": result["id"],
                "relevance_score": round(1 - result.get("distance", 0), 3),
                **result["metadata"]
            }
            formatted_results.append(formatted_result)
        
        return JSONResponse({
            "message": f"Found {len(results)} matching listings",
            "preferences": preferences,
            "count": len(results),
            "results": formatted_results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching listings: {str(e)}")


@app.get("/api/database-info")
async def get_database_info():
    """Get information about the vector database"""
    try:
        info = vector_store.get_collection_info()
        return JSONResponse(info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database info: {str(e)}")


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
