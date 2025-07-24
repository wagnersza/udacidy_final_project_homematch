"""
Vector Store Module for HomeMatch
Handles ChromaDB operations for storing and retrieving real estate listings
"""

import os
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VectorStore:
    """Manages vector database operations for real estate listings"""
    
    def __init__(self, persist_directory: str = None):
        """Initialize ChromaDB client and collection"""
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", "./database/chroma_db")
        
        # Ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection for listings
        self.collection_name = "real_estate_listings"
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"📂 Connected to existing collection: {self.collection_name}")
        except Exception:
            # Create new collection with embedding function
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Real estate listings with embeddings"}
            )
            print(f"🆕 Created new collection: {self.collection_name}")
    
    def _prepare_listing_text(self, listing: Dict[str, Any]) -> str:
        """Prepare listing text for embedding generation"""
        # Combine all relevant text fields for embedding
        text_parts = [
            f"Neighborhood: {listing.get('neighborhood', '')}",
            f"Price: {listing.get('price', '')}",
            f"Bedrooms: {listing.get('bedrooms', '')}",
            f"Bathrooms: {listing.get('bathrooms', '')}",
            f"Size: {listing.get('house_size', '')}",
            f"Description: {listing.get('description', '')}",
            f"Area: {listing.get('neighborhood_description', '')}"
        ]
        
        return " ".join(text_parts)
    
    def add_listing(self, listing: Dict[str, Any]) -> bool:
        """Add a single listing to the vector database"""
        try:
            listing_id = listing.get('id', f"listing_{hash(str(listing))}")
            listing_text = self._prepare_listing_text(listing)
            
            # Create metadata (ChromaDB metadata must be strings, ints, floats, or bools)
            metadata = {
                "neighborhood": str(listing.get('neighborhood', '')),
                "price": str(listing.get('price', '')),
                "bedrooms": int(listing.get('bedrooms', 0)) if isinstance(listing.get('bedrooms'), (int, float)) else 0,
                "bathrooms": int(listing.get('bathrooms', 0)) if isinstance(listing.get('bathrooms'), (int, float)) else 0,
                "house_size": str(listing.get('house_size', '')),
                "generated_at": str(listing.get('generated_at', ''))
            }
            
            # Add to collection (ChromaDB will automatically generate embeddings)
            self.collection.add(
                documents=[listing_text],
                metadatas=[metadata],
                ids=[listing_id]
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding listing {listing_id}: {e}")
            return False
    
    def add_multiple_listings(self, listings: List[Dict[str, Any]]) -> int:
        """Add multiple listings to the vector database"""
        print(f"🔄 Adding {len(listings)} listings to vector database...")
        
        success_count = 0
        batch_size = 10  # Process in batches to avoid memory issues
        
        for i in range(0, len(listings), batch_size):
            batch = listings[i:i + batch_size]
            
            try:
                # Prepare batch data
                ids = []
                documents = []
                metadatas = []
                
                for listing in batch:
                    listing_id = listing.get('id', f"listing_{hash(str(listing))}")
                    listing_text = self._prepare_listing_text(listing)
                    
                    metadata = {
                        "neighborhood": str(listing.get('neighborhood', '')),
                        "price": str(listing.get('price', '')),
                        "bedrooms": int(listing.get('bedrooms', 0)) if isinstance(listing.get('bedrooms'), (int, float)) else 0,
                        "bathrooms": int(listing.get('bathrooms', 0)) if isinstance(listing.get('bathrooms'), (int, float)) else 0,
                        "house_size": str(listing.get('house_size', '')),
                        "generated_at": str(listing.get('generated_at', ''))
                    }
                    
                    ids.append(listing_id)
                    documents.append(listing_text)
                    metadatas.append(metadata)
                
                # Add batch to collection
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                success_count += len(batch)
                print(f"✅ Added batch {i//batch_size + 1}: {len(batch)} listings")
                
            except Exception as e:
                print(f"❌ Error adding batch {i//batch_size + 1}: {e}")
        
        print(f"🎉 Successfully added {success_count}/{len(listings)} listings to vector database!")
        return success_count
    
    def search_listings(self, query: str, n_results: int = 5, metadata_filters: Optional[Dict[str, Any]] = None, 
                       where_document: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for listings based on semantic similarity with advanced filtering"""
        try:
            print(f"🔍 Searching for: '{query}'")
            if metadata_filters:
                print(f"🎯 With metadata filters: {metadata_filters}")
            if where_document:
                print(f"📄 With document filters: {where_document}")
            
            # Build query parameters
            query_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            
            # Add metadata filters if provided
            if metadata_filters:
                query_params["where"] = metadata_filters
            
            # Add document content filters if provided
            if where_document:
                query_params["where_document"] = where_document
            
            # Perform semantic search with filters
            results = self.collection.query(**query_params)
            
            # Format results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'id': results['ids'][0][i],
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None,
                        'similarity_score': 1 - results['distances'][0][i] if 'distances' in results and results['distances'][0][i] is not None else 0.0
                    }
                    formatted_results.append(result)
            
            print(f"📋 Found {len(formatted_results)} matching listings")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching listings: {e}")
            return []
    
    def search_with_preferences(self, preferences_obj, n_results: int = 5, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Advanced search combining semantic similarity with preference-based filtering
        
        Args:
            preferences_obj: UserPreferences object from preference_processor
            n_results: Number of results to return
            semantic_weight: Weight for semantic similarity vs metadata matching (0.0-1.0)
            
        Returns:
            List of ranked and scored listings
        """
        try:
            from .preference_processor import UserPreferences, PropertyType, PriceRange
            
            print(f"🎯 Searching with structured preferences")
            
            # Build semantic query from preferences
            semantic_query_parts = []
            
            # Add location preferences
            if preferences_obj.neighborhoods:
                semantic_query_parts.extend(preferences_obj.neighborhoods)
            if preferences_obj.location_keywords:
                semantic_query_parts.extend(preferences_obj.location_keywords)
            
            # Add lifestyle preferences
            if preferences_obj.lifestyle_keywords:
                semantic_query_parts.extend(preferences_obj.lifestyle_keywords)
            
            # Add amenity preferences
            if preferences_obj.amenities:
                semantic_query_parts.extend(preferences_obj.amenities)
            
            # Add property type
            if preferences_obj.property_type != PropertyType.ANY:
                semantic_query_parts.append(preferences_obj.property_type.value)
            
            # Create semantic query
            semantic_query = " ".join(semantic_query_parts) if semantic_query_parts else preferences_obj.raw_input
            
            # Build metadata filters
            metadata_filters = {}
            
            # Build metadata filters - ChromaDB requires specific structure
            metadata_filters = {}
            filter_conditions = []
            
            # Bedroom filters
            if preferences_obj.min_bedrooms is not None and preferences_obj.max_bedrooms is not None:
                if preferences_obj.min_bedrooms == preferences_obj.max_bedrooms:
                    # Exact match
                    filter_conditions.append({"bedrooms": {"$eq": preferences_obj.min_bedrooms}})
                else:
                    # Range - use $and to combine conditions
                    filter_conditions.append({"bedrooms": {"$gte": preferences_obj.min_bedrooms}})
                    filter_conditions.append({"bedrooms": {"$lte": preferences_obj.max_bedrooms}})
            elif preferences_obj.min_bedrooms is not None:
                filter_conditions.append({"bedrooms": {"$gte": preferences_obj.min_bedrooms}})
            elif preferences_obj.max_bedrooms is not None:
                filter_conditions.append({"bedrooms": {"$lte": preferences_obj.max_bedrooms}})
            
            # Bathroom filters
            if preferences_obj.min_bathrooms is not None and preferences_obj.max_bathrooms is not None:
                if preferences_obj.min_bathrooms == preferences_obj.max_bathrooms:
                    # Exact match
                    filter_conditions.append({"bathrooms": {"$eq": preferences_obj.min_bathrooms}})
                else:
                    # Range - use $and to combine conditions
                    filter_conditions.append({"bathrooms": {"$gte": preferences_obj.min_bathrooms}})
                    filter_conditions.append({"bathrooms": {"$lte": preferences_obj.max_bathrooms}})
            elif preferences_obj.min_bathrooms is not None:
                filter_conditions.append({"bathrooms": {"$gte": preferences_obj.min_bathrooms}})
            elif preferences_obj.max_bathrooms is not None:
                filter_conditions.append({"bathrooms": {"$lte": preferences_obj.max_bathrooms}})
            
            # Combine all filter conditions using $and if there are multiple
            if len(filter_conditions) == 1:
                metadata_filters = filter_conditions[0]
            elif len(filter_conditions) > 1:
                metadata_filters = {"$and": filter_conditions}            # Build document content filters for amenities and neighborhoods
            document_filters = {}
            
            # Add neighborhood filters
            if preferences_obj.neighborhoods:
                # Use OR logic for neighborhoods
                neighborhood_conditions = []
                for neighborhood in preferences_obj.neighborhoods:
                    neighborhood_conditions.append({"$contains": neighborhood.lower()})
                if len(neighborhood_conditions) == 1:
                    document_filters = neighborhood_conditions[0]
                else:
                    document_filters["$or"] = neighborhood_conditions
            
            # Perform the search
            results = self.search_listings(
                query=semantic_query,
                n_results=n_results * 2,  # Get more results for re-ranking
                metadata_filters=metadata_filters if metadata_filters else None,
                where_document=document_filters if document_filters else None
            )
            
            # Re-rank results with preference matching
            ranked_results = self._rank_with_preferences(results, preferences_obj, semantic_weight)
            
            return ranked_results[:n_results]
            
        except Exception as e:
            print(f"❌ Error in preference-based search: {e}")
            # Fallback to basic search
            return self.search_listings(preferences_obj.raw_input, n_results)
    
    def _rank_with_preferences(self, results: List[Dict[str, Any]], preferences_obj, semantic_weight: float) -> List[Dict[str, Any]]:
        """
        Re-rank search results based on preference matching
        
        Args:
            results: Initial search results
            preferences_obj: UserPreferences object
            semantic_weight: Weight for semantic vs preference matching
            
        Returns:
            Re-ranked results with composite scores
        """
        try:
            from .preference_processor import PropertyType, PriceRange
            
            for result in results:
                metadata = result.get('metadata', {})
                content = result.get('content', '').lower()
                
                # Start with semantic similarity score
                semantic_score = result.get('similarity_score', 0.0)
                
                # Calculate preference matching score
                preference_score = 0.0
                total_weight = 0.0
                
                # Location preference matching
                location_weight = preferences_obj.priority_weights.get('location', 1.0)
                location_score = 0.0
                
                if preferences_obj.neighborhoods:
                    for neighborhood in preferences_obj.neighborhoods:
                        if neighborhood.lower() in content:
                            location_score += 0.5
                    location_score = min(location_score, 1.0)
                
                if preferences_obj.location_keywords:
                    for keyword in preferences_obj.location_keywords:
                        if keyword.lower() in content:
                            location_score += 0.3
                    location_score = min(location_score, 1.0)
                
                preference_score += location_score * location_weight
                total_weight += location_weight
                
                # Amenity preference matching
                amenity_weight = preferences_obj.priority_weights.get('amenities', 0.6)
                amenity_score = 0.0
                
                if preferences_obj.amenities:
                    matched_amenities = 0
                    for amenity in preferences_obj.amenities:
                        if amenity.lower() in content:
                            matched_amenities += 1
                    amenity_score = matched_amenities / len(preferences_obj.amenities)
                
                preference_score += amenity_score * amenity_weight
                total_weight += amenity_weight
                
                # Lifestyle preference matching
                if preferences_obj.lifestyle_keywords:
                    lifestyle_score = 0.0
                    for keyword in preferences_obj.lifestyle_keywords:
                        if keyword.lower() in content:
                            lifestyle_score += 0.5
                    lifestyle_score = min(lifestyle_score, 1.0)
                    
                    preference_score += lifestyle_score * 0.4  # Fixed weight for lifestyle
                    total_weight += 0.4
                
                # Size preference matching (if size info is available)
                size_weight = preferences_obj.priority_weights.get('size', 0.8)
                size_score = 1.0  # Default to neutral
                
                # Try to extract size from metadata or content
                house_size_str = metadata.get('house_size', '')
                if house_size_str and any(char.isdigit() for char in house_size_str):
                    try:
                        # Extract numeric value from size string
                        import re
                        size_match = re.search(r'(\d{1,4}(?:,\d{3})*)', house_size_str.replace(',', ''))
                        if size_match:
                            property_size = int(size_match.group(1))
                            
                            # Score based on size preferences
                            if preferences_obj.min_size and property_size < preferences_obj.min_size:
                                size_score = 0.3
                            elif preferences_obj.max_size and property_size > preferences_obj.max_size:
                                size_score = 0.3
                            else:
                                size_score = 1.0
                    except:
                        pass
                
                preference_score += size_score * size_weight
                total_weight += size_weight
                
                # Normalize preference score
                if total_weight > 0:
                    preference_score /= total_weight
                
                # Calculate composite score
                composite_score = (semantic_score * semantic_weight + 
                                 preference_score * (1 - semantic_weight))
                
                # Add scores to result
                result['preference_score'] = preference_score
                result['composite_score'] = composite_score
                result['ranking_details'] = {
                    'semantic_score': semantic_score,
                    'preference_score': preference_score,
                    'location_score': location_score if 'location_score' in locals() else 0.0,
                    'amenity_score': amenity_score if 'amenity_score' in locals() else 0.0,
                    'size_score': size_score if 'size_score' in locals() else 1.0
                }
            
            # Sort by composite score
            results.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in preference ranking: {e}")
            return results
    
    def get_listing_by_id(self, listing_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific listing by ID"""
        try:
            results = self.collection.get(ids=[listing_id])
            
            if results['documents']:
                return {
                    'id': results['ids'][0],
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving listing {listing_id}: {e}")
            return None
    
    def get_all_listings(self) -> List[Dict[str, Any]]:
        """Retrieve all listings from the database"""
        try:
            results = self.collection.get()
            
            formatted_results = []
            for i, doc in enumerate(results['documents']):
                result = {
                    'id': results['ids'][i],
                    'content': doc,
                    'metadata': results['metadatas'][i]
                }
                formatted_results.append(result)
            
            print(f"📋 Retrieved {len(formatted_results)} total listings")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error retrieving all listings: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
            return {}
    
    def delete_listing(self, listing_id: str) -> bool:
        """Delete a listing from the database"""
        try:
            self.collection.delete(ids=[listing_id])
            print(f"🗑️ Deleted listing: {listing_id}")
            return True
        except Exception as e:
            print(f"❌ Error deleting listing {listing_id}: {e}")
            return False
    
    def clear_all_listings(self) -> bool:
        """Clear all listings from the database (use with caution!)"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Real estate listings with embeddings"}
            )
            print("🧹 Cleared all listings from database")
            return True
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
            return False


def main():
    """Test the vector store functionality"""
    print("🧪 Testing Vector Store functionality...")
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Get collection info
    info = vector_store.get_collection_info()
    print(f"📊 Collection info: {info}")
    
    # Test with sample listing
    sample_listing = {
        "id": "test_001",
        "neighborhood": "Green Oaks",
        "price": "$800,000",
        "bedrooms": 3,
        "bathrooms": 2,
        "house_size": "2,000 sqft",
        "description": "Beautiful eco-friendly home with solar panels and modern amenities",
        "neighborhood_description": "Quiet, environmentally-conscious community with parks and bike paths",
        "generated_at": "2025-07-23"
    }
    
    # Add sample listing
    success = vector_store.add_listing(sample_listing)
    print(f"✅ Added sample listing: {success}")
    
    # Test search
    results = vector_store.search_listings("eco-friendly house with solar panels", n_results=3)
    print(f"🔍 Search results: {len(results)}")
    
    if results:
        print("📋 First result:")
        print(f"   ID: {results[0]['id']}")
        print(f"   Neighborhood: {results[0]['metadata']['neighborhood']}")
        print(f"   Price: {results[0]['metadata']['price']}")


if __name__ == "__main__":
    main()
