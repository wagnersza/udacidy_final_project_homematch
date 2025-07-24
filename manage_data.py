"""
Management script for HomeMatch data initialization
Run this script to generate initial data and set up the vector database
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_generator import DataGenerator
from modules.vector_store import VectorStore

load_dotenv()


async def initialize_data():
    """Initialize the HomeMatch system with sample data"""
    print("🚀 Initializing HomeMatch Data System...")
    print("=" * 50)
    
    # Initialize components
    data_generator = DataGenerator()
    vector_store = VectorStore()
    
    # Check current state
    print("\n📊 Current System State:")
    db_info = vector_store.get_collection_info()
    print(f"   Vector Database: {db_info.get('count', 0)} listings")
    
    existing_listings = data_generator.load_listings_from_file()
    print(f"   JSON File: {len(existing_listings)} listings")
    
    # Generate new data if needed
    if len(existing_listings) < 10:
        print("\n🏠 Generating new real estate listings...")
        
        try:
            # Generate 12 diverse listings
            listings = await data_generator.generate_multiple_listings(12)
            
            # Save to file
            filepath = data_generator.save_listings_to_file(listings)
            print(f"💾 Saved to: {filepath}")
            
            # Add to vector database
            added_count = vector_store.add_multiple_listings(listings)
            print(f"🗄️  Added {added_count} listings to vector database")
            
        except Exception as e:
            print(f"❌ Error during data generation: {e}")
            print("Using fallback data...")
            
            # Create fallback listings
            fallback_listings = [data_generator._create_fallback_listing(i) for i in range(12)]
            data_generator.save_listings_to_file(fallback_listings)
            vector_store.add_multiple_listings(fallback_listings)
    
    else:
        print(f"\n✅ Data already exists. Using {len(existing_listings)} existing listings.")
        
        # Ensure vector database is populated
        if db_info.get('count', 0) == 0:
            print("🔄 Populating vector database with existing listings...")
            vector_store.add_multiple_listings(existing_listings)
    
    # Final status check
    print("\n📈 Final System Status:")
    final_db_info = vector_store.get_collection_info()
    final_listings = data_generator.load_listings_from_file()
    
    print(f"   ✅ Vector Database: {final_db_info.get('count', 0)} listings")
    print(f"   ✅ JSON File: {len(final_listings)} listings")
    print(f"   ✅ Database Location: {final_db_info.get('persist_directory', 'N/A')}")
    
    # Show sample data
    if final_listings:
        print("\n🏠 Sample Listing:")
        sample = final_listings[0]
        print(f"   🏷️  {sample.get('id', 'N/A')}")
        print(f"   🏘️  {sample.get('neighborhood', 'N/A')}")
        print(f"   💰 {sample.get('price', 'N/A')}")
        print(f"   🛏️  {sample.get('bedrooms', 'N/A')} bed, {sample.get('bathrooms', 'N/A')} bath")
        print(f"   📏 {sample.get('house_size', 'N/A')}")
        
        description = sample.get('description', '')
        print(f"   📝 {description[:80]}{'...' if len(description) > 80 else ''}")
    
    # Test search functionality
    print("\n🔍 Testing Search Functionality:")
    test_queries = [
        "family home with garden",
        "modern apartment downtown",
        "house with good schools"
    ]
    
    for query in test_queries:
        results = vector_store.search_listings(query, n_results=2)
        print(f"   Query: '{query}' → {len(results)} results")
    
    print("\n🎉 Data initialization complete!")
    print("🚀 You can now start the FastAPI server with: python app.py")


async def clear_data():
    """Clear all existing data (use with caution!)"""
    print("⚠️  WARNING: This will clear all existing data!")
    confirmation = input("Type 'YES' to confirm: ")
    
    if confirmation == "YES":
        vector_store = VectorStore()
        vector_store.clear_all_listings()
        
        # Remove JSON file
        if os.path.exists("listings.json"):
            os.remove("listings.json")
            print("🗑️ Removed listings.json")
        
        print("🧹 All data cleared!")
    else:
        print("❌ Operation cancelled")


def show_help():
    """Show available commands"""
    print("HomeMatch Data Management")
    print("=" * 30)
    print("Available commands:")
    print("  init    - Initialize data (generate listings and populate database)")
    print("  clear   - Clear all existing data")
    print("  help    - Show this help message")
    print("\nUsage: python manage_data.py [command]")


async def main():
    """Main function"""
    if len(sys.argv) < 2:
        command = "init"  # Default command
    else:
        command = sys.argv[1].lower()
    
    if command == "init":
        await initialize_data()
    elif command == "clear":
        await clear_data()
    elif command == "help":
        show_help()
    else:
        print(f"❌ Unknown command: {command}")
        show_help()


if __name__ == "__main__":
    asyncio.run(main())
