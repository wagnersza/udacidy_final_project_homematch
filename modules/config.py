"""
HomeMatch Configuration Management
Centralized configuration for the HomeMatch application with environment-specific settings
"""

import os
import logging
from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings with validation and environment variable support"""
    
    # API Configuration
    app_name: str = Field(default="HomeMatch", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development/staging/production)")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=True, description="Auto-reload on code changes")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI base URL")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model to use")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", description="OpenAI embedding model")
    openai_max_tokens: int = Field(default=1000, description="Maximum tokens for OpenAI responses")
    openai_temperature: float = Field(default=0.7, description="OpenAI temperature setting")
    
    # Database Configuration
    chroma_persist_directory: str = Field(default="./database/chroma_db", description="ChromaDB persistence directory")
    chroma_collection_name: str = Field(default="real_estate_listings", description="ChromaDB collection name")
    
    # Data Configuration
    listings_file_path: str = Field(default="listings.json", description="Path to listings JSON file")
    min_listings_count: int = Field(default=10, description="Minimum number of listings to generate")
    max_search_results: int = Field(default=20, description="Maximum search results to return")
    
    # Performance Configuration
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent requests")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Feature Flags
    enable_synthetic_data_generation: bool = Field(default=True, description="Enable synthetic data generation")
    enable_personalization: bool = Field(default=True, description="Enable description personalization")
    enable_advanced_search: bool = Field(default=True, description="Enable advanced search features")
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_rate_limiting: bool = Field(default=False, description="Enable rate limiting")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
        
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"
        
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": self.log_format,
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.FileHandler",
                    "filename": self.log_file or f"{self.app_name.lower()}.log",
                },
            },
            "root": {
                "level": self.log_level,
                "handlers": ["default"] + (["file"] if self.log_file else []),
            },
            "loggers": {
                "homewatch": {
                    "level": self.log_level,
                    "handlers": ["default"] + (["file"] if self.log_file else []),
                    "propagate": False,
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["default"],
                    "propagate": False,
                },
            },
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance"""
    return settings


def validate_configuration() -> Dict[str, Any]:
    """Validate configuration and return status"""
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }
    
    # Check required API keys
    if not settings.openai_api_key or settings.openai_api_key == "your_api_key_here":
        validation_results["errors"].append("OpenAI API key not configured properly")
        validation_results["valid"] = False
    
    # Check directory permissions
    try:
        os.makedirs(settings.chroma_persist_directory, exist_ok=True)
        validation_results["info"]["chroma_directory"] = f"Accessible: {settings.chroma_persist_directory}"
    except Exception as e:
        validation_results["errors"].append(f"Cannot access ChromaDB directory: {e}")
        validation_results["valid"] = False
    
    # Check environment-specific settings
    if settings.is_production():
        if settings.debug:
            validation_results["warnings"].append("Debug mode enabled in production")
        if settings.reload:
            validation_results["warnings"].append("Auto-reload enabled in production")
    
    # Performance validation
    if settings.openai_max_tokens > 4000:
        validation_results["warnings"].append("High token limit may impact performance and costs")
    
    validation_results["info"]["environment"] = settings.environment
    validation_results["info"]["features_enabled"] = {
        "synthetic_data": settings.enable_synthetic_data_generation,
        "personalization": settings.enable_personalization,
        "advanced_search": settings.enable_advanced_search,
        "caching": settings.enable_caching,
        "rate_limiting": settings.enable_rate_limiting
    }
    
    return validation_results
