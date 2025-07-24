"""
HomeMatch - Personalized Real Estate Recommendation System
A RAG-based application that provides personalized real estate recommendations
using Large Language Models and vector databases.
"""

import os
import time
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, Request, Form, HTTPException, Depends
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
from modules.config import get_settings, validate_configuration, Settings
from modules.monitoring import get_logger, get_health_checker, timing_decorator, HomeMatchLogger
from modules.testing import QualityAssuranceTester

# Load environment variables
load_dotenv()

# Initialize settings and validation
settings = get_settings()
config_validation = validate_configuration()

# Setup logging
logger = get_logger("homewatch")
if not config_validation["valid"]:
    logger.error("Configuration validation failed", extra={"errors": config_validation["errors"]})
    for error in config_validation["errors"]:
        logger.error(f"Config error: {error}")

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Personalized Real Estate Recommendation System using RAG architecture",
    version=settings.app_version,
    docs_url="/docs" if settings.is_development() else None,
    redoc_url="/redoc" if settings.is_development() else None,
    openapi_url="/api/openapi.json" if settings.is_development() else None
)

# Initialize modules with enhanced logging
logger.info("Initializing application modules")
data_generator = DataGenerator()
vector_store = VectorStore()
preference_processor = PreferenceProcessor()
description_personalizer = DescriptionPersonalizer()
qa_tester = QualityAssuranceTester()

# Initialize health checker and register health checks
health_checker = get_health_checker()

def check_openai_connection() -> Dict[str, Any]:
    """Health check for OpenAI API connection"""
    try:
        # Simple test to verify API key is valid
        if not settings.openai_api_key or settings.openai_api_key == "your_api_key_here":
            return {
                "status": "unhealthy",
                "message": "OpenAI API key not configured"
            }
        return {
            "status": "healthy",
            "message": "OpenAI API configuration appears valid"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"OpenAI connection check failed: {str(e)}"
        }

def check_vector_database() -> Dict[str, Any]:
    """Health check for vector database"""
    try:
        # Test vector store connection
        listings = vector_store.get_all_listings()
        return {
            "status": "healthy",
            "message": f"Vector database accessible with {len(listings)} listings",
            "listing_count": len(listings)
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "message": f"Vector database check failed: {str(e)}"
        }

def check_data_files() -> Dict[str, Any]:
    """Health check for data files"""
    try:
        listings = data_generator.load_listings_from_file()
        if listings:
            return {
                "status": "healthy",
                "message": f"Data files accessible with {len(listings)} listings",
                "file_listing_count": len(listings)
            }
        else:
            return {
                "status": "degraded",
                "message": "No listings found in data files"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Data file check failed: {str(e)}"
        }

# Register health checks
health_checker.register_check("openai", check_openai_connection)
health_checker.register_check("vector_database", check_vector_database)
health_checker.register_check("data_files", check_data_files)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for configuration (backward compatibility)
OPENAI_API_KEY = settings.openai_api_key
OPENAI_BASE_URL = settings.openai_base_url
CHROMA_PERSIST_DIRECTORY = settings.chroma_persist_directory

logger.info("Application initialization completed")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with user preference form"""
    logger.info("Home page accessed")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    logger.debug("Basic health check requested")
    return {"status": "healthy", "message": "HomeMatch API is running", "timestamp": time.time()}


@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check with all component status"""
    logger.info("Detailed health check requested")
    
    try:
        health_results = health_checker.run_all_checks()
        
        # Add configuration validation results
        health_results["configuration"] = config_validation
        
        # Add application metrics
        health_results["metrics"] = logger.metrics.get_metrics_summary()
        
        # Set appropriate HTTP status code
        status_code = 200
        if health_results["overall_status"] == "unhealthy":
            status_code = 503
        elif health_results["overall_status"] == "degraded":
            status_code = 200  # Still functional
            
        return JSONResponse(content=health_results, status_code=status_code)
        
    except Exception as e:
        logger.error("Health check failed", error=e)
        return JSONResponse(
            content={
                "overall_status": "error",
                "error": str(e),
                "timestamp": time.time()
            },
            status_code=500
        )


@app.get("/api/config")
async def get_config():
    """Get API configuration information"""
    logger.debug("Configuration information requested")
    return {
        "api_version": settings.app_version,
        "environment": settings.environment,
        "features": {
            "ai_powered_search": settings.enable_advanced_search,
            "personalization": settings.enable_personalization,
            "vector_search": True,
            "synthetic_data_generation": settings.enable_synthetic_data_generation,
            "caching": settings.enable_caching,
            "rate_limiting": settings.enable_rate_limiting
        },
        "endpoints": {
            "search": "/api/search",
            "personalize": "/api/personalize", 
            "generate_listings": "/api/generate-listings",
            "preferences": "/api/preferences",
            "health": "/health",
            "detailed_health": "/health/detailed",
            "metrics": "/api/metrics",
            "testing": "/api/test"
        },
        "limits": {
            "max_search_results": settings.max_search_results,
            "request_timeout": settings.request_timeout,
            "max_concurrent_requests": settings.max_concurrent_requests
        }
    }


@app.get("/api/metrics")
async def get_metrics():
    """Get application performance metrics"""
    logger.debug("Metrics requested")
    
    try:
        metrics = logger.metrics.get_metrics_summary()
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error("Failed to retrieve metrics", error=e)
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")


@app.post("/api/test")
async def run_comprehensive_tests():
    """Run comprehensive system tests"""
    logger.info("Comprehensive testing requested")
    
    try:
        test_results = qa_tester.run_comprehensive_tests()
        
        # Log test results summary
        summary = test_results["test_summary"]
        logger.info(
            f"Testing completed: {summary['passed']}/{summary['total_tests']} passed",
            extra={"test_summary": summary}
        )
        
        return JSONResponse(content=test_results)
        
    except Exception as e:
        logger.error("Comprehensive testing failed", error=e)
        raise HTTPException(status_code=500, detail=f"Error running tests: {str(e)}")


@app.get("/api/test/search")
async def test_search_functionality():
    """Test search functionality specifically"""
    logger.info("Search functionality testing requested")
    
    try:
        search_results = qa_tester.test_search_functionality()
        return JSONResponse(content=search_results)
    except Exception as e:
        logger.error("Search testing failed", error=e)
        raise HTTPException(status_code=500, detail=f"Error testing search: {str(e)}")


@app.get("/api/test/performance") 
async def test_performance():
    """Test system performance"""
    logger.info("Performance testing requested")
    
    try:
        performance_results = qa_tester.test_performance()
        return JSONResponse(content=performance_results)
    except Exception as e:
        logger.error("Performance testing failed", error=e)
        raise HTTPException(status_code=500, detail=f"Error testing performance: {str(e)}")


@app.get("/api/listings")
@timing_decorator("get_listings", logger)
async def get_listings():
    """Get all available listings"""
    logger.info("Listings requested")
    
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
        
        logger.info(f"Retrieved {len(listings)} listings")
        return JSONResponse({
            "count": len(listings),
            "listings": listings
        })
        
    except Exception as e:
        logger.error("Failed to retrieve listings", error=e)
        raise HTTPException(status_code=500, detail=f"Error retrieving listings: {str(e)}")


@app.post("/api/generate-listings")
@timing_decorator("generate_listings", logger)
async def generate_listings(count: int = 12):
    """Generate new synthetic real estate listings"""
    logger.info(f"Generating {count} new listings")
    
    try:
        print(f"🏠 Generating {count} new listings...")
        
        # Generate listings
        listings = await data_generator.generate_multiple_listings(count)
        
        # Save to file
        filepath = data_generator.save_listings_to_file(listings)
        
        # Add to vector database
        added_count = vector_store.add_multiple_listings(listings)
        
        logger.info(f"Successfully generated {len(listings)} listings")
        logger.metrics.increment_counter("listings_generated", {"count": str(len(listings))})
        
        return JSONResponse({
            "message": f"Successfully generated {len(listings)} listings",
            "count": len(listings),
            "saved_to_file": filepath,
            "added_to_vector_db": added_count,
            "listings": listings[:3]  # Return first 3 as sample
        })
        
    except Exception as e:
        logger.error("Failed to generate listings", error=e)
        logger.metrics.increment_counter("listings_generation_errors")
        raise HTTPException(status_code=500, detail=f"Error generating listings: {str(e)}")


@app.post("/api/search")
@timing_decorator("search_listings", logger)
async def search_listings(request: Request, preferences: str = Form(...), 
                         use_advanced_search: bool = Form(default=True),
                         max_results: int = Form(default=5)):
    """Enhanced search for listings based on user preferences with intelligent processing"""
    logger.info(f"Search request: '{preferences[:100]}...'")
    
    try:
        if not preferences.strip():
            raise HTTPException(status_code=400, detail="Preferences cannot be empty")
        
        # Validate max_results parameter
        max_results = min(max_results, settings.max_search_results)
        
        print(f"🔍 Processing search request: '{preferences}'")
        
        # Record search metrics
        logger.metrics.increment_counter("search_requests")
        search_start_time = time.time()
        
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
    # Setup logging configuration
    import logging.config
    log_config = settings.get_log_config()
    logging.config.dictConfig(log_config)
    
    # Verify configuration
    logger.info("Starting HomeMatch application...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"OpenAI Base URL: {settings.openai_base_url}")
    logger.info(f"Chroma DB Directory: {settings.chroma_persist_directory}")
    
    # Log configuration validation results
    if config_validation["valid"]:
        logger.info("Configuration validation passed")
    else:
        logger.warning("Configuration validation failed")
        for error in config_validation["errors"]:
            logger.error(f"Config error: {error}")
        for warning in config_validation["warnings"]:
            logger.warning(f"Config warning: {warning}")
    
    # Log feature flags
    logger.info("Feature flags:", extra={
        "synthetic_data": settings.enable_synthetic_data_generation,
        "personalization": settings.enable_personalization,
        "advanced_search": settings.enable_advanced_search,
        "caching": settings.enable_caching,
        "rate_limiting": settings.enable_rate_limiting
    })
    
    # Start the server
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload and settings.is_development(),
        log_level=settings.log_level.lower(),
        access_log=True
    )
