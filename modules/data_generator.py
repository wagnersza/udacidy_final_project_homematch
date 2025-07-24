"""
Data Generator Module for HomeMatch
Generates synthetic real estate listings using OpenAI's GPT model
"""

import json
import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()


class DataGenerator:
    """Generates synthetic real estate listings using OpenAI API"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        
    def generate_listing_prompt(self, listing_index: int) -> str:
        """Generate a structured prompt for creating real estate listings"""
        
        neighborhoods = [
            "Green Oaks", "Sunset Hills", "Riverside Gardens", "Downtown District",
            "Maple Grove", "Ocean View", "Historic District", "Tech Valley",
            "Suburban Heights", "Arts Quarter", "University District", "Pine Ridge"
        ]
        
        price_ranges = [
            "400,000-600,000", "600,000-800,000", "800,000-1,200,000", 
            "300,000-500,000", "1,200,000-1,800,000"
        ]
        
        prompt = f"""Generate a realistic real estate listing with the following structure. Make it unique and varied:

Neighborhood: {neighborhoods[listing_index % len(neighborhoods)]}
Price: ${price_ranges[listing_index % len(price_ranges)].split('-')[0]} - ${price_ranges[listing_index % len(price_ranges)].split('-')[1]}
Bedrooms: [2-5 bedrooms]
Bathrooms: [1-4 bathrooms] 
House Size: [1,200-3,500 sqft]

Description: [Write a compelling 3-4 sentence description highlighting unique features, architectural style, recent updates, and lifestyle benefits. Make it realistic and appealing.]

Neighborhood Description: [Write 2-3 sentences about the neighborhood, including local amenities, community feel, nearby attractions, schools, transportation, and what makes it special.]

Format the response as valid JSON with this exact structure:
{{
    "neighborhood": "neighborhood name",
    "price": "$XXX,XXX",
    "bedrooms": X,
    "bathrooms": X,
    "house_size": "X,XXX sqft",
    "description": "property description here",
    "neighborhood_description": "neighborhood details here"
}}

Make this listing unique and realistic. Vary the architectural styles, amenities, and neighborhood characteristics."""

        return prompt
    
    async def generate_single_listing(self, listing_index: int) -> Dict[str, Any]:
        """Generate a single real estate listing"""
        try:
            prompt = self.generate_listing_prompt(listing_index)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional real estate content generator. Create realistic, varied, and appealing property listings in valid JSON format."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.8
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content.strip()
            
            # Clean up the response to ensure valid JSON
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            listing_data = json.loads(content)
            
            # Add metadata
            listing_data["id"] = f"listing_{listing_index + 1:03d}"
            listing_data["generated_at"] = "2025-07-23"
            
            print(f"✅ Generated listing {listing_index + 1}: {listing_data['neighborhood']}")
            return listing_data
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error for listing {listing_index + 1}: {e}")
            # Return a fallback listing
            return self._create_fallback_listing(listing_index)
        except Exception as e:
            print(f"❌ Error generating listing {listing_index + 1}: {e}")
            return self._create_fallback_listing(listing_index)
    
    def _create_fallback_listing(self, listing_index: int) -> Dict[str, Any]:
        """Create a fallback listing if API generation fails"""
        neighborhoods = [
            "Green Oaks", "Sunset Hills", "Riverside Gardens", "Downtown District",
            "Maple Grove", "Ocean View", "Historic District", "Tech Valley"
        ]
        
        return {
            "id": f"listing_{listing_index + 1:03d}",
            "neighborhood": neighborhoods[listing_index % len(neighborhoods)],
            "price": "$750,000",
            "bedrooms": 3,
            "bathrooms": 2,
            "house_size": "2,000 sqft",
            "description": "Beautiful family home with modern amenities and spacious layout. Perfect for comfortable living with updated kitchen and bathrooms.",
            "neighborhood_description": "Quiet, family-friendly neighborhood with excellent schools and convenient access to shopping and transportation.",
            "generated_at": "2025-07-23"
        }
    
    async def generate_multiple_listings(self, count: int = 12) -> List[Dict[str, Any]]:
        """Generate multiple real estate listings"""
        print(f"🏠 Generating {count} real estate listings...")
        
        listings = []
        for i in range(count):
            try:
                listing = await self.generate_single_listing(i)
                listings.append(listing)
                
                # Add a small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Failed to generate listing {i + 1}: {e}")
                # Add fallback listing
                listings.append(self._create_fallback_listing(i))
        
        print(f"✅ Successfully generated {len(listings)} listings!")
        return listings
    
    def save_listings_to_file(self, listings: List[Dict[str, Any]], filename: str = "listings.json") -> str:
        """Save listings to JSON file"""
        try:
            filepath = os.path.join(os.getcwd(), filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(listings, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved {len(listings)} listings to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving listings: {e}")
            return ""
    
    def load_listings_from_file(self, filename: str = "listings.json") -> List[Dict[str, Any]]:
        """Load listings from JSON file"""
        try:
            filepath = os.path.join(os.getcwd(), filename)
            
            if not os.path.exists(filepath):
                print(f"📁 File {filepath} not found")
                return []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                listings = json.load(f)
            
            print(f"📂 Loaded {len(listings)} listings from {filepath}")
            return listings
            
        except Exception as e:
            print(f"❌ Error loading listings: {e}")
            return []


async def main():
    """Main function to test the data generator"""
    generator = DataGenerator()
    
    # Check if API key is configured
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_api_key_here":
        print("⚠️  Warning: OpenAI API key not configured. Using fallback listings.")
        # Generate fallback listings
        listings = [generator._create_fallback_listing(i) for i in range(12)]
    else:
        # Generate listings using OpenAI API
        listings = await generator.generate_multiple_listings(12)
    
    # Save to file
    generator.save_listings_to_file(listings)
    
    # Display sample
    print("\n📋 Sample listing:")
    if listings:
        sample = listings[0]
        print(f"🏠 {sample['neighborhood']} - {sample['price']}")
        print(f"🛏️  {sample['bedrooms']} bed, {sample['bathrooms']} bath, {sample['house_size']}")
        print(f"📝 {sample['description'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
