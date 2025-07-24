"""
Description Personalizer Module for HomeMatch
Handles OpenAI-powered listing description enhancement based on user preferences
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import openai
from dotenv import load_dotenv
import re
import time

# Load environment variables
load_dotenv()


@dataclass
class PersonalizationResult:
    """Result of description personalization"""
    original_description: str
    personalized_description: str
    highlights: List[str]
    preference_matches: List[str]
    personalization_score: float
    processing_time: float
    fallback_used: bool = False
    error_message: str = ""


class DescriptionPersonalizer:
    """Personalizes real estate listing descriptions based on user preferences"""
    
    def __init__(self):
        """Initialize the description personalizer"""
        self.openai_client = None
        
        # Initialize OpenAI client if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_api_key_here":
            try:
                self.openai_client = openai.OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized for description personalization")
            except Exception as e:
                print(f"⚠️ Failed to initialize OpenAI client: {e}")
        else:
            print("⚠️ OpenAI API key not found, using fallback personalization")
        
        # Template system for consistent description structure
        self.description_templates = {
            "family_friendly": {
                "opening": "Perfect for families,",
                "highlights": ["spacious layout", "family-friendly neighborhood", "excellent schools nearby"],
                "closing": "This home offers the ideal environment for raising a family."
            },
            "luxury": {
                "opening": "Indulge in luxury with",
                "highlights": ["premium finishes", "high-end amenities", "exclusive location"],
                "closing": "Experience sophisticated living at its finest."
            },
            "modern": {
                "opening": "Embrace contemporary living in",
                "highlights": ["modern design", "state-of-the-art features", "sleek aesthetics"],
                "closing": "Where style meets functionality."
            },
            "eco_friendly": {
                "opening": "Sustainable living meets comfort in",
                "highlights": ["energy-efficient features", "eco-friendly materials", "green spaces"],
                "closing": "Live responsibly without compromising on comfort."
            },
            "urban": {
                "opening": "City living at its best,",
                "highlights": ["downtown location", "walkable neighborhood", "vibrant community"],
                "closing": "Everything you need is just steps away."
            },
            "default": {
                "opening": "Discover your next home in",
                "highlights": ["desirable location", "well-maintained property", "great value"],
                "closing": "Schedule your viewing today."
            }
        }
        
        # Fallback enhancement patterns
        self.enhancement_patterns = {
            "amenities": {
                "pool": ["dive into relaxation", "perfect for summer entertaining", "your private oasis"],
                "garage": ["secure parking", "additional storage space", "convenience and protection"],
                "garden": ["outdoor sanctuary", "green space for relaxation", "perfect for gardening enthusiasts"],
                "fireplace": ["cozy ambiance", "warmth and charm", "perfect for intimate evenings"],
                "kitchen": ["culinary haven", "heart of the home", "perfect for entertaining"]
            },
            "lifestyle": {
                "quiet": ["peaceful retreat", "serene environment", "escape the hustle and bustle"],
                "family": ["family-oriented", "child-friendly", "perfect for growing families"],
                "luxury": ["sophisticated", "elegant", "premium quality"],
                "modern": ["contemporary", "up-to-date", "cutting-edge design"]
            }
        }
    
    def personalize_description(self, listing: Dict[str, Any], preferences_obj, 
                              use_llm: bool = True) -> PersonalizationResult:
        """
        Personalize a listing description based on user preferences
        
        Args:
            listing: Property listing with metadata and description
            preferences_obj: UserPreferences object
            use_llm: Whether to use OpenAI LLM for personalization
            
        Returns:
            PersonalizationResult with personalized content
        """
        start_time = time.time()
        
        original_description = listing.get('description', '')
        if not original_description:
            # Try to get description from content or metadata
            original_description = listing.get('content', listing.get('metadata', {}).get('description', ''))
        
        result = PersonalizationResult(
            original_description=original_description,
            personalized_description=original_description,
            highlights=[],
            preference_matches=[],
            personalization_score=0.0,
            processing_time=0.0
        )
        
        try:
            if use_llm and self.openai_client:
                result = self._personalize_with_llm(listing, preferences_obj, result)
            else:
                result = self._personalize_with_templates(listing, preferences_obj, result)
                result.fallback_used = True
                
        except Exception as e:
            print(f"❌ Error in personalization, using fallback: {e}")
            result = self._personalize_with_templates(listing, preferences_obj, result)
            result.fallback_used = True
            result.error_message = str(e)
        
        result.processing_time = time.time() - start_time
        return result
    
    def _personalize_with_llm(self, listing: Dict[str, Any], preferences_obj, 
                             result: PersonalizationResult) -> PersonalizationResult:
        """Use OpenAI LLM for advanced personalization"""
        
        # Prepare context about the listing
        metadata = listing.get('metadata', {})
        listing_context = {
            "neighborhood": metadata.get('neighborhood', 'Unknown'),
            "price": metadata.get('price', 'Unknown'),
            "bedrooms": metadata.get('bedrooms', 'Unknown'),
            "bathrooms": metadata.get('bathrooms', 'Unknown'),
            "size": metadata.get('house_size', 'Unknown'),
            "original_description": result.original_description
        }
        
        # Prepare user preferences context
        preferences_context = {
            "neighborhoods": preferences_obj.neighborhoods,
            "lifestyle_keywords": preferences_obj.lifestyle_keywords,
            "amenities": preferences_obj.amenities,
            "property_type": preferences_obj.property_type.value,
            "price_range": preferences_obj.price_range.value,
            "bedrooms": f"{preferences_obj.min_bedrooms}-{preferences_obj.max_bedrooms}",
            "raw_preferences": preferences_obj.raw_input
        }
        
        prompt = f"""
        You are a professional real estate copywriter specializing in personalized property descriptions.
        Your task is to rewrite a property listing to match a specific buyer's preferences and lifestyle.
        
        PROPERTY DETAILS:
        - Neighborhood: {listing_context['neighborhood']}
        - Price: {listing_context['price']}
        - Bedrooms: {listing_context['bedrooms']}
        - Bathrooms: {listing_context['bathrooms']}
        - Size: {listing_context['size']}
        
        ORIGINAL DESCRIPTION:
        "{listing_context['original_description']}"
        
        BUYER PREFERENCES:
        - Preferred neighborhoods: {preferences_context['neighborhoods']}
        - Lifestyle preferences: {preferences_context['lifestyle_keywords']}
        - Desired amenities: {preferences_context['amenities']}
        - Property type: {preferences_context['property_type']}
        - Price range: {preferences_context['price_range']}
        - Original request: "{preferences_context['raw_preferences']}"
        
        INSTRUCTIONS:
        1. Rewrite the description to appeal specifically to this buyer's preferences
        2. Highlight features that match their lifestyle and needs
        3. Use language that resonates with their preferences
        4. Maintain factual accuracy - DO NOT add features that aren't mentioned
        5. Keep the description engaging and professional
        6. Length should be 100-200 words
        
        Provide your response in the following JSON format:
        {{
            "personalized_description": "The rewritten description",
            "key_highlights": ["highlight 1", "highlight 2", "highlight 3"],
            "preference_matches": ["matched preference 1", "matched preference 2"],
            "personalization_reasoning": "Brief explanation of personalization strategy"
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            llm_result = json.loads(response.choices[0].message.content)
            
            result.personalized_description = llm_result.get('personalized_description', result.original_description)
            result.highlights = llm_result.get('key_highlights', [])
            result.preference_matches = llm_result.get('preference_matches', [])
            
            # Calculate personalization score based on preference matches
            result.personalization_score = self._calculate_personalization_score(
                result.personalized_description, preferences_obj
            )
            
            print(f"✅ LLM personalization completed (score: {result.personalization_score:.2f})")
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing LLM response: {e}")
            raise
        except Exception as e:
            print(f"❌ Error in LLM personalization: {e}")
            raise
        
        return result
    
    def _personalize_with_templates(self, listing: Dict[str, Any], preferences_obj, 
                                   result: PersonalizationResult) -> PersonalizationResult:
        """Use template-based personalization as fallback"""
        
        metadata = listing.get('metadata', {})
        content = (result.original_description + " " + str(metadata)).lower()
        
        # Determine the best template based on preferences
        template_key = "default"
        
        if any(keyword in preferences_obj.lifestyle_keywords for keyword in ["family", "family-friendly", "schools"]):
            template_key = "family_friendly"
        elif any(keyword in preferences_obj.lifestyle_keywords for keyword in ["luxury", "premium", "high-end"]):
            template_key = "luxury"
        elif any(keyword in preferences_obj.lifestyle_keywords for keyword in ["modern", "contemporary"]):
            template_key = "modern"
        elif any(keyword in preferences_obj.lifestyle_keywords for keyword in ["eco", "green", "sustainable"]):
            template_key = "eco_friendly"
        elif any(keyword in preferences_obj.lifestyle_keywords for keyword in ["urban", "downtown", "city"]):
            template_key = "urban"
        
        template = self.description_templates[template_key]
        
        # Build personalized description
        personalized_parts = []
        
        # Opening
        personalized_parts.append(template["opening"])
        
        # Add original description with enhancements
        enhanced_description = result.original_description
        
        # Enhance based on matched amenities
        for amenity in preferences_obj.amenities:
            if amenity.lower() in content:
                result.preference_matches.append(f"Has preferred amenity: {amenity}")
                if amenity in self.enhancement_patterns["amenities"]:
                    enhancement = self.enhancement_patterns["amenities"][amenity][0]
                    enhanced_description = enhanced_description.replace(
                        amenity, f"{amenity} ({enhancement})"
                    )
        
        # Enhance based on lifestyle keywords
        for keyword in preferences_obj.lifestyle_keywords:
            if keyword in content:
                result.preference_matches.append(f"Matches lifestyle preference: {keyword}")
        
        personalized_parts.append(enhanced_description)
        
        # Add highlights
        relevant_highlights = []
        for highlight in template["highlights"]:
            if any(word in content for word in highlight.split()):
                relevant_highlights.append(highlight)
        
        if relevant_highlights:
            result.highlights = relevant_highlights
            personalized_parts.append("Key features include: " + ", ".join(relevant_highlights) + ".")
        
        # Closing
        personalized_parts.append(template["closing"])
        
        result.personalized_description = " ".join(personalized_parts)
        
        # Calculate personalization score
        result.personalization_score = self._calculate_personalization_score(
            result.personalized_description, preferences_obj
        )
        
        print(f"✅ Template personalization completed (score: {result.personalization_score:.2f})")
        
        return result
    
    def _calculate_personalization_score(self, description: str, preferences_obj) -> float:
        """Calculate how well the description matches user preferences"""
        
        description_lower = description.lower()
        score = 0.0
        total_checks = 0
        
        # Check neighborhood matches
        if preferences_obj.neighborhoods:
            total_checks += 1
            neighborhood_matches = sum(1 for neighborhood in preferences_obj.neighborhoods 
                                     if neighborhood.lower() in description_lower)
            score += min(neighborhood_matches / len(preferences_obj.neighborhoods), 1.0)
        
        # Check amenity matches
        if preferences_obj.amenities:
            total_checks += 1
            amenity_matches = sum(1 for amenity in preferences_obj.amenities 
                                if amenity.lower() in description_lower)
            score += min(amenity_matches / len(preferences_obj.amenities), 1.0)
        
        # Check lifestyle keyword matches
        if preferences_obj.lifestyle_keywords:
            total_checks += 1
            lifestyle_matches = sum(1 for keyword in preferences_obj.lifestyle_keywords 
                                  if keyword.lower() in description_lower)
            score += min(lifestyle_matches / len(preferences_obj.lifestyle_keywords), 1.0)
        
        # Check property type match
        if preferences_obj.property_type.value != "any":
            total_checks += 1
            if preferences_obj.property_type.value.lower() in description_lower:
                score += 1.0
        
        return score / total_checks if total_checks > 0 else 0.0
    
    def personalize_multiple_listings(self, listings: List[Dict[str, Any]], preferences_obj, 
                                    use_llm: bool = True, max_concurrent: int = 3) -> List[PersonalizationResult]:
        """
        Personalize multiple listings with rate limiting
        
        Args:
            listings: List of property listings
            preferences_obj: UserPreferences object
            use_llm: Whether to use OpenAI LLM
            max_concurrent: Maximum concurrent LLM requests
            
        Returns:
            List of PersonalizationResult objects
        """
        results = []
        
        for i, listing in enumerate(listings):
            try:
                print(f"🎨 Personalizing listing {i+1}/{len(listings)}")
                
                # Add rate limiting for LLM calls
                if use_llm and self.openai_client and i > 0:
                    time.sleep(0.5)  # Rate limiting
                
                result = self.personalize_description(listing, preferences_obj, use_llm)
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error personalizing listing {i+1}: {e}")
                # Create error result
                error_result = PersonalizationResult(
                    original_description=listing.get('description', ''),
                    personalized_description=listing.get('description', ''),
                    highlights=[],
                    preference_matches=[],
                    personalization_score=0.0,
                    processing_time=0.0,
                    fallback_used=True,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    def validate_personalized_content(self, result: PersonalizationResult) -> bool:
        """Validate that personalized content maintains factual integrity"""
        
        try:
            original_words = set(result.original_description.lower().split())
            personalized_words = set(result.personalized_description.lower().split())
            
            # Check for completely different content (potential hallucination)
            common_words = original_words.intersection(personalized_words)
            
            # If less than 30% of words are common, flag as potentially problematic
            if len(common_words) / max(len(original_words), 1) < 0.3:
                print("⚠️ Personalized content may have deviated too far from original")
                return False
            
            # Check for obvious fabrications (this is basic, could be enhanced)
            fabrication_indicators = [
                "just renovated", "newly built", "award-winning", "featured in",
                "celebrity owned", "historic", "landmark"
            ]
            
            original_lower = result.original_description.lower()
            personalized_lower = result.personalized_description.lower()
            
            for indicator in fabrication_indicators:
                if indicator in personalized_lower and indicator not in original_lower:
                    print(f"⚠️ Potential fabrication detected: '{indicator}'")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating personalized content: {e}")
            return False


def main():
    """Test the description personalizer functionality"""
    print("🧪 Testing Description Personalizer functionality...")
    
    # Import preferences for testing
    from preference_processor import PreferenceProcessor, UserPreferences
    
    personalizer = DescriptionPersonalizer()
    processor = PreferenceProcessor()
    
    # Test listing
    test_listing = {
        "id": "test_001",
        "metadata": {
            "neighborhood": "Green Oaks",
            "price": "$800,000",
            "bedrooms": 3,
            "bathrooms": 2,
            "house_size": "2,000 sqft"
        },
        "description": "Beautiful home featuring an open floor plan with hardwood floors, updated kitchen with granite countertops, spacious master bedroom, and a large backyard perfect for entertaining. Located in a quiet neighborhood with excellent schools nearby."
    }
    
    # Test preferences
    user_input = "Looking for a family-friendly home with good schools, must have a backyard for kids"
    preferences = processor.process_preferences(user_input, use_llm=False)
    
    print(f"\nOriginal description: {test_listing['description']}")
    print(f"User preferences: {user_input}")
    
    # Test personalization
    result = personalizer.personalize_description(test_listing, preferences, use_llm=False)
    
    print(f"\nPersonalized description: {result.personalized_description}")
    print(f"Highlights: {result.highlights}")
    print(f"Preference matches: {result.preference_matches}")
    print(f"Personalization score: {result.personalization_score:.2f}")
    print(f"Processing time: {result.processing_time:.3f}s")
    print(f"Fallback used: {result.fallback_used}")


if __name__ == "__main__":
    main()
