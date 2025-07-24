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
from modules.preference_processor import PreferenceProcessor, UserPreferences
from modules.description_personalizer import DescriptionPersonalizer

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="HomeMatch",
    description="Personalized Real Estate Recommendation System using RAG architecture",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json"
)

# Initialize modules
data_generator = DataGenerator()
vector_store = VectorStore()
preference_processor = PreferenceProcessor()
description_personalizer = DescriptionPersonalizer()

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


@app.get("/api/config")
async def get_config():
    """Get API configuration information"""
    return {
        "api_version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "features": {
            "ai_powered_search": True,
            "personalization": True,
            "vector_search": True,
            "synthetic_data_generation": True
        },
        "endpoints": {
            "search": "/api/search",
            "personalize": "/api/personalize", 
            "generate_listings": "/api/generate-listings",
            "preferences": "/api/preferences"
        }
    }


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
async def search_listings(request: Request, preferences: str = Form(...), 
                         use_advanced_search: bool = Form(default=True),
                         max_results: int = Form(default=5)):
    """Enhanced search for listings based on user preferences with intelligent processing"""
    try:
        if not preferences.strip():
            raise HTTPException(status_code=400, detail="Preferences cannot be empty")
        
        print(f"🔍 Processing search request: '{preferences}'")
        
        if use_advanced_search:
            # Use enhanced preference-based search
            try:
                # Process preferences into structured format
                user_preferences = preference_processor.process_preferences(preferences, use_llm=True)
                
                print(f"🎯 Structured preferences extracted:")
                print(f"  - Property type: {user_preferences.property_type.value}")
                print(f"  - Bedrooms: {user_preferences.min_bedrooms}-{user_preferences.max_bedrooms}")
                print(f"  - Price range: {user_preferences.price_range.value}")
                print(f"  - Neighborhoods: {user_preferences.neighborhoods}")
                print(f"  - Amenities: {user_preferences.amenities}")
                print(f"  - Lifestyle: {user_preferences.lifestyle_keywords}")
                
                # Perform preference-based search
                results = vector_store.search_with_preferences(
                    user_preferences, 
                    n_results=max_results,
                    semantic_weight=0.7
                )
                
                # Format results with enhanced scoring
                formatted_results = []
                for result in results:
                    formatted_result = {
                        "id": result["id"],
                        "relevance_score": round(result.get("similarity_score", 0), 3),
                        "preference_score": round(result.get("preference_score", 0), 3),
                        "composite_score": round(result.get("composite_score", 0), 3),
                        "ranking_details": result.get("ranking_details", {}),
                        **result["metadata"]
                    }
                    formatted_results.append(formatted_result)
                
                return JSONResponse({
                    "message": f"Found {len(results)} matching listings using advanced search",
                    "search_type": "preference_based",
                    "preferences": preferences,
                    "structured_preferences": user_preferences.to_dict(),
                    "count": len(results),
                    "results": formatted_results
                })
                
            except Exception as e:
                print(f"⚠️ Advanced search failed, falling back to basic search: {e}")
                use_advanced_search = False
        
        if not use_advanced_search:
            # Fallback to basic semantic search
            results = vector_store.search_listings(preferences, n_results=max_results)
            
            if not results:
                # If no results from vector search, return all listings as fallback
                all_listings = data_generator.load_listings_from_file()
                return JSONResponse({
                    "message": "No semantic matches found. Showing all available listings.",
                    "search_type": "fallback",
                    "preferences": preferences,
                    "count": len(all_listings),
                    "results": all_listings[:max_results]
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
                "message": f"Found {len(results)} matching listings using basic search",
                "search_type": "semantic",
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
async def personalize_listing(request: Request, 
                             listing_id: str = Form(...),
                             preferences: str = Form(...),
                             use_llm: bool = Form(default=True)):
    """Personalize listing descriptions based on user preferences"""
    try:
        print(f"🎨 Personalizing listing {listing_id} for preferences: '{preferences}'")
        
        # Get the listing
        listing = vector_store.get_listing_by_id(listing_id)
        if not listing:
            raise HTTPException(status_code=404, detail=f"Listing {listing_id} not found")
        
        # Process user preferences
        user_preferences = preference_processor.process_preferences(preferences, use_llm=use_llm)
        
        # Personalize the description
        personalization_result = description_personalizer.personalize_description(
            listing, user_preferences, use_llm=use_llm
        )
        
        # Validate the personalized content
        is_valid = description_personalizer.validate_personalized_content(personalization_result)
        
        response_data = {
            "listing_id": listing_id,
            "original_description": personalization_result.original_description,
            "personalized_description": personalization_result.personalized_description,
            "highlights": personalization_result.highlights,
            "preference_matches": personalization_result.preference_matches,
            "personalization_score": round(personalization_result.personalization_score, 3),
            "processing_time": round(personalization_result.processing_time, 3),
            "fallback_used": personalization_result.fallback_used,
            "content_validated": is_valid,
            "metadata": listing.get("metadata", {}),
            "user_preferences": user_preferences.to_dict()
        }
        
        if personalization_result.error_message:
            response_data["error_message"] = personalization_result.error_message
        
        return JSONResponse(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error personalizing listing: {str(e)}")


@app.post("/api/preferences")
async def process_preferences(request: Request, 
                             preferences: str = Form(...),
                             use_llm: bool = Form(default=True)):
    """Process and structure user preferences"""
    try:
        print(f"🧠 Processing preferences: '{preferences}'")
        
        # Process the preferences
        user_preferences = preference_processor.process_preferences(preferences, use_llm=use_llm)
        
        return JSONResponse({
            "message": "Preferences processed successfully",
            "raw_input": preferences,
            "structured_preferences": user_preferences.to_dict(),
            "search_filters": user_preferences.to_search_filters(),
            "processing_summary": {
                "neighborhoods_found": len(user_preferences.neighborhoods),
                "amenities_found": len(user_preferences.amenities),
                "lifestyle_keywords": len(user_preferences.lifestyle_keywords),
                "property_type": user_preferences.property_type.value,
                "price_range": user_preferences.price_range.value
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing preferences: {str(e)}")


@app.post("/api/personalized-descriptions")
async def get_personalized_descriptions(request: Request,
                                       preferences: str = Form(...),
                                       listing_ids: str = Form(...),  # Comma-separated IDs
                                       use_llm: bool = Form(default=True)):
    """Get personalized descriptions for multiple listings"""
    try:
        # Parse listing IDs
        ids = [id.strip() for id in listing_ids.split(',') if id.strip()]
        if not ids:
            raise HTTPException(status_code=400, detail="No listing IDs provided")
        
        print(f"🎨 Personalizing {len(ids)} listings for preferences: '{preferences}'")
        
        # Process user preferences
        user_preferences = preference_processor.process_preferences(preferences, use_llm=use_llm)
        
        # Get listings
        listings = []
        for listing_id in ids:
            listing = vector_store.get_listing_by_id(listing_id)
            if listing:
                listings.append(listing)
            else:
                print(f"⚠️ Listing {listing_id} not found")
        
        if not listings:
            raise HTTPException(status_code=404, detail="No valid listings found")
        
        # Personalize descriptions
        personalization_results = description_personalizer.personalize_multiple_listings(
            listings, user_preferences, use_llm=use_llm
        )
        
        # Format response
        results = []
        for i, result in enumerate(personalization_results):
            listing = listings[i]
            is_valid = description_personalizer.validate_personalized_content(result)
            
            result_data = {
                "listing_id": listing.get("id"),
                "original_description": result.original_description,
                "personalized_description": result.personalized_description,
                "highlights": result.highlights,
                "preference_matches": result.preference_matches,
                "personalization_score": round(result.personalization_score, 3),
                "processing_time": round(result.processing_time, 3),
                "fallback_used": result.fallback_used,
                "content_validated": is_valid,
                "metadata": listing.get("metadata", {})
            }
            
            if result.error_message:
                result_data["error_message"] = result.error_message
            
            results.append(result_data)
        
        return JSONResponse({
            "message": f"Personalized {len(results)} listings",
            "user_preferences": user_preferences.to_dict(),
            "total_processing_time": round(sum(r.processing_time for r in personalization_results), 3),
            "average_personalization_score": round(sum(r.personalization_score for r in personalization_results) / len(personalization_results), 3),
            "results": results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error personalizing descriptions: {str(e)}")


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
