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
    
    def search_listings(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for listings based on semantic similarity"""
        try:
            print(f"🔍 Searching for: '{query}'")
            
            # Perform semantic search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'id': results['ids'][0][i],
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    }
                    formatted_results.append(result)
            
            print(f"📋 Found {len(formatted_results)} matching listings")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching listings: {e}")
            return []
    
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
