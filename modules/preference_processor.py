"""
Preference Processing Module for HomeMatch
Handles intelligent user preference parsing, extraction, and validation
"""

import re
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class PropertyType(Enum):
    """Enumeration of property types"""
    HOUSE = "house"
    APARTMENT = "apartment"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"
    ANY = "any"


class PriceRange(Enum):
    """Enumeration of price ranges"""
    UNDER_500K = "under_500k"
    FROM_500K_TO_750K = "500k_to_750k"
    FROM_750K_TO_1M = "750k_to_1m"
    FROM_1M_TO_1_5M = "1m_to_1_5m"
    ABOVE_1_5M = "above_1_5m"
    ANY = "any"


@dataclass
class UserPreferences:
    """Structured representation of user preferences"""
    
    # Location preferences
    neighborhoods: List[str] = None
    location_keywords: List[str] = None
    
    # Property specifications
    property_type: PropertyType = PropertyType.ANY
    min_bedrooms: Optional[int] = None
    max_bedrooms: Optional[int] = None
    min_bathrooms: Optional[int] = None
    max_bathrooms: Optional[int] = None
    
    # Price preferences
    price_range: PriceRange = PriceRange.ANY
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    
    # Size preferences
    min_size: Optional[int] = None  # in square feet
    max_size: Optional[int] = None
    
    # Lifestyle and amenities
    amenities: List[str] = None
    lifestyle_keywords: List[str] = None
    
    # Priorities (weighted importance)
    priority_weights: Dict[str, float] = None
    
    # Raw input for reference
    raw_input: str = ""
    
    def __post_init__(self):
        """Initialize default values"""
        if self.neighborhoods is None:
            self.neighborhoods = []
        if self.location_keywords is None:
            self.location_keywords = []
        if self.amenities is None:
            self.amenities = []
        if self.lifestyle_keywords is None:
            self.lifestyle_keywords = []
        if self.priority_weights is None:
            self.priority_weights = {
                "location": 1.0,
                "price": 1.0,
                "size": 0.8,
                "amenities": 0.6,
                "property_type": 0.7
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert enums to their values
        result['property_type'] = self.property_type.value
        result['price_range'] = self.price_range.value
        return result
    
    def to_search_filters(self) -> Dict[str, Any]:
        """Convert to ChromaDB-compatible search filters"""
        filters = {}
        
        # Property type filter
        if self.property_type != PropertyType.ANY:
            # This would need to be adapted based on how property types are stored
            pass
        
        # Bedroom filters
        if self.min_bedrooms is not None:
            filters["bedrooms"] = {"$gte": self.min_bedrooms}
        if self.max_bedrooms is not None:
            if "bedrooms" in filters:
                filters["bedrooms"]["$lte"] = self.max_bedrooms
            else:
                filters["bedrooms"] = {"$lte": self.max_bedrooms}
        
        # Bathroom filters
        if self.min_bathrooms is not None:
            filters["bathrooms"] = {"$gte": self.min_bathrooms}
        if self.max_bathrooms is not None:
            if "bathrooms" in filters:
                filters["bathrooms"]["$lte"] = self.max_bathrooms
            else:
                filters["bathrooms"] = {"$lte": self.max_bathrooms}
        
        return filters


class PreferenceProcessor:
    """Processes and extracts structured preferences from natural language input"""
    
    def __init__(self):
        """Initialize the preference processor"""
        self.openai_client = None
        
        # Initialize OpenAI client if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_api_key_here":
            self.openai_client = openai.OpenAI(api_key=api_key)
        
        # Common patterns for regex-based extraction
        self.price_patterns = {
            r'\$(\d{1,3}(?:,\d{3})*(?:k|K))': lambda m: int(m.group(1).replace(',', '').replace('k', '').replace('K', '')) * 1000,
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:m|M))': lambda m: int(float(m.group(1).replace(',', '').replace('m', '').replace('M', '')) * 1000000),
            r'\$(\d{1,3}(?:,\d{3})*)': lambda m: int(m.group(1).replace(',', ''))
        }
        
        self.bedroom_patterns = [
            r'(\d+)\s*(?:bed|bedroom|br)s?',
            r'(\d+)\s*(?:bed|bedroom|br)s?\s*(?:room|rooms)?',
        ]
        
        self.bathroom_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:bath|bathroom|ba)s?',
            r'(\d+(?:\.\d+)?)\s*(?:bath|bathroom|ba)s?\s*(?:room|rooms)?',
        ]
        
        self.size_patterns = [
            r'(\d{1,4}(?:,\d{3})*)\s*(?:sq\s*ft|sqft|square\s*feet)',
            r'(\d{1,4}(?:,\d{3})*)\s*(?:sf)',
        ]
        
        # Common amenities and lifestyle keywords
        self.amenity_keywords = [
            'pool', 'gym', 'parking', 'garage', 'garden', 'yard', 'patio', 'balcony',
            'fireplace', 'hardwood', 'granite', 'stainless', 'air conditioning', 'heating',
            'dishwasher', 'washer', 'dryer', 'elevator', 'security', 'doorman',
            'pet-friendly', 'furnished', 'unfurnished'
        ]
        
        self.lifestyle_keywords = [
            'quiet', 'family-friendly', 'modern', 'luxury', 'budget', 'eco-friendly',
            'walkable', 'commuter', 'urban', 'suburban', 'rural', 'downtown',
            'near schools', 'near transit', 'near parks'
        ]
    
    def process_preferences(self, user_input: str, use_llm: bool = True) -> UserPreferences:
        """
        Process user input and extract structured preferences
        
        Args:
            user_input: Natural language description of preferences
            use_llm: Whether to use LLM for advanced extraction
            
        Returns:
            UserPreferences object with extracted preferences
        """
        preferences = UserPreferences(raw_input=user_input)
        
        # Start with regex-based extraction
        self._extract_with_regex(user_input, preferences)
        
        # Enhance with LLM if available
        if use_llm and self.openai_client:
            try:
                self._extract_with_llm(user_input, preferences)
            except Exception as e:
                print(f"⚠️ LLM extraction failed, using regex-only: {e}")
        
        # Validate and normalize
        self._validate_and_normalize(preferences)
        
        return preferences
    
    def _extract_with_regex(self, user_input: str, preferences: UserPreferences):
        """Extract preferences using regex patterns"""
        
        # Convert to lowercase for pattern matching
        text = user_input.lower()
        
        # Extract price information
        prices = []
        for pattern, converter in self.price_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                prices.append(converter(re.match(pattern, f"${match}")))
        
        if prices:
            prices.sort()
            preferences.min_price = prices[0] if len(prices) == 1 else prices[0]
            preferences.max_price = prices[-1] if len(prices) > 1 else None
        
        # Extract bedroom information
        for pattern in self.bedroom_patterns:
            matches = re.findall(pattern, text)
            if matches:
                bedrooms = int(matches[0])
                if not preferences.min_bedrooms:
                    preferences.min_bedrooms = bedrooms
                preferences.max_bedrooms = bedrooms
                break
        
        # Extract bathroom information
        for pattern in self.bathroom_patterns:
            matches = re.findall(pattern, text)
            if matches:
                bathrooms = float(matches[0])
                if not preferences.min_bathrooms:
                    preferences.min_bathrooms = bathrooms
                preferences.max_bathrooms = bathrooms
                break
        
        # Extract size information
        for pattern in self.size_patterns:
            matches = re.findall(pattern, text)
            if matches:
                size = int(matches[0].replace(',', ''))
                if not preferences.min_size:
                    preferences.min_size = size
                preferences.max_size = size
                break
        
        # Extract amenities
        for amenity in self.amenity_keywords:
            if amenity.lower() in text:
                preferences.amenities.append(amenity)
        
        # Extract lifestyle keywords
        for keyword in self.lifestyle_keywords:
            if keyword.lower() in text:
                preferences.lifestyle_keywords.append(keyword)
        
        # Extract property type
        if any(word in text for word in ['house', 'single family']):
            preferences.property_type = PropertyType.HOUSE
        elif any(word in text for word in ['apartment', 'apt']):
            preferences.property_type = PropertyType.APARTMENT
        elif 'condo' in text:
            preferences.property_type = PropertyType.CONDO
        elif 'townhouse' in text:
            preferences.property_type = PropertyType.TOWNHOUSE
    
    def _extract_with_llm(self, user_input: str, preferences: UserPreferences):
        """Enhance extraction using OpenAI LLM"""
        
        prompt = f"""
        Extract structured real estate preferences from the following user input. 
        Enhance and complement the existing extracted data, don't overwrite unless clearly contradicted.
        
        User Input: "{user_input}"
        
        Current extracted data:
        - Property type: {preferences.property_type.value}
        - Bedrooms: {preferences.min_bedrooms}-{preferences.max_bedrooms}
        - Bathrooms: {preferences.min_bathrooms}-{preferences.max_bathrooms}
        - Price range: ${preferences.min_price}-${preferences.max_price}
        - Size: {preferences.min_size}-{preferences.max_size} sq ft
        - Amenities: {preferences.amenities}
        - Lifestyle: {preferences.lifestyle_keywords}
        
        Please provide a JSON response with the following structure:
        {{
            "neighborhoods": ["list of mentioned neighborhoods or areas"],
            "location_keywords": ["list of location-related preferences"],
            "property_type": "house|apartment|condo|townhouse|any",
            "min_bedrooms": null or number,
            "max_bedrooms": null or number,
            "min_bathrooms": null or number,
            "max_bathrooms": null or number,
            "price_range": "under_500k|500k_to_750k|750k_to_1m|1m_to_1_5m|above_1_5m|any",
            "min_price": null or number,
            "max_price": null or number,
            "min_size": null or number,
            "max_size": null or number,
            "amenities": ["list of mentioned amenities"],
            "lifestyle_keywords": ["list of lifestyle preferences"],
            "priority_weights": {{
                "location": 0.0-1.0,
                "price": 0.0-1.0,
                "size": 0.0-1.0,
                "amenities": 0.0-1.0,
                "property_type": 0.0-1.0
            }}
        }}
        
        Focus on:
        1. Neighborhoods, cities, or areas mentioned
        2. Lifestyle preferences (quiet, family-friendly, walkable, etc.)
        3. Specific amenities or features
        4. Price ranges and priorities
        5. Any implied preferences from context
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Update preferences with LLM results
            if result.get('neighborhoods'):
                preferences.neighborhoods.extend(result['neighborhoods'])
            if result.get('location_keywords'):
                preferences.location_keywords.extend(result['location_keywords'])
            if result.get('amenities'):
                preferences.amenities.extend(result['amenities'])
            if result.get('lifestyle_keywords'):
                preferences.lifestyle_keywords.extend(result['lifestyle_keywords'])
            
            # Update numeric fields if not already set or if LLM provides better values
            for field in ['min_bedrooms', 'max_bedrooms', 'min_bathrooms', 'max_bathrooms', 
                         'min_price', 'max_price', 'min_size', 'max_size']:
                if result.get(field) is not None and getattr(preferences, field) is None:
                    setattr(preferences, field, result[field])
            
            # Update property type if LLM provides a specific type
            if result.get('property_type') and result['property_type'] != 'any':
                try:
                    preferences.property_type = PropertyType(result['property_type'])
                except ValueError:
                    pass
            
            # Update priority weights
            if result.get('priority_weights'):
                preferences.priority_weights.update(result['priority_weights'])
                
        except Exception as e:
            print(f"❌ Error in LLM extraction: {e}")
    
    def _validate_and_normalize(self, preferences: UserPreferences):
        """Validate and normalize extracted preferences"""
        
        # Remove duplicates
        preferences.neighborhoods = list(set(preferences.neighborhoods))
        preferences.location_keywords = list(set(preferences.location_keywords))
        preferences.amenities = list(set(preferences.amenities))
        preferences.lifestyle_keywords = list(set(preferences.lifestyle_keywords))
        
        # Validate numeric ranges
        if preferences.min_bedrooms and preferences.max_bedrooms:
            if preferences.min_bedrooms > preferences.max_bedrooms:
                preferences.min_bedrooms, preferences.max_bedrooms = preferences.max_bedrooms, preferences.min_bedrooms
        
        if preferences.min_bathrooms and preferences.max_bathrooms:
            if preferences.min_bathrooms > preferences.max_bathrooms:
                preferences.min_bathrooms, preferences.max_bathrooms = preferences.max_bathrooms, preferences.min_bathrooms
        
        if preferences.min_price and preferences.max_price:
            if preferences.min_price > preferences.max_price:
                preferences.min_price, preferences.max_price = preferences.max_price, preferences.min_price
        
        if preferences.min_size and preferences.max_size:
            if preferences.min_size > preferences.max_size:
                preferences.min_size, preferences.max_size = preferences.max_size, preferences.min_size
        
        # Set price range based on numeric values
        if preferences.min_price or preferences.max_price:
            max_price = preferences.max_price or preferences.min_price
            if max_price < 500000:
                preferences.price_range = PriceRange.UNDER_500K
            elif max_price < 750000:
                preferences.price_range = PriceRange.FROM_500K_TO_750K
            elif max_price < 1000000:
                preferences.price_range = PriceRange.FROM_750K_TO_1M
            elif max_price < 1500000:
                preferences.price_range = PriceRange.FROM_1M_TO_1_5M
            else:
                preferences.price_range = PriceRange.ABOVE_1_5M
        
        # Normalize priority weights
        total_weight = sum(preferences.priority_weights.values())
        if total_weight > 0:
            for key in preferences.priority_weights:
                preferences.priority_weights[key] /= total_weight
    
    def merge_preferences(self, pref1: UserPreferences, pref2: UserPreferences) -> UserPreferences:
        """Merge two preference objects, with pref2 taking priority"""
        
        merged = UserPreferences()
        
        # Merge lists
        merged.neighborhoods = list(set(pref1.neighborhoods + pref2.neighborhoods))
        merged.location_keywords = list(set(pref1.location_keywords + pref2.location_keywords))
        merged.amenities = list(set(pref1.amenities + pref2.amenities))
        merged.lifestyle_keywords = list(set(pref1.lifestyle_keywords + pref2.lifestyle_keywords))
        
        # Use pref2 values where available, otherwise pref1
        for field in ['property_type', 'min_bedrooms', 'max_bedrooms', 'min_bathrooms', 
                     'max_bathrooms', 'price_range', 'min_price', 'max_price', 
                     'min_size', 'max_size']:
            value2 = getattr(pref2, field)
            value1 = getattr(pref1, field)
            
            if value2 is not None and (field.startswith('property_type') or field.startswith('price_range') or value2 != 0):
                setattr(merged, field, value2)
            else:
                setattr(merged, field, value1)
        
        # Merge priority weights
        merged.priority_weights = {**pref1.priority_weights, **pref2.priority_weights}
        
        # Use latest raw input
        merged.raw_input = pref2.raw_input or pref1.raw_input
        
        return merged


def main():
    """Test the preference processor functionality"""
    print("🧪 Testing Preference Processor functionality...")
    
    processor = PreferenceProcessor()
    
    # Test cases
    test_inputs = [
        "I'm looking for a 3 bedroom house under $800,000 in a quiet neighborhood with a backyard",
        "Need 2-3 bed apartment downtown, budget around $500k, must have parking and gym",
        "Family home in Green Oaks, 4+ bedrooms, good schools nearby, pool preferred",
        "Modern condo, 2br 2ba, luxury building, $750k-$1M range, walkable area"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {user_input}")
        
        preferences = processor.process_preferences(user_input, use_llm=False)
        print(f"Extracted preferences:")
        print(f"  Property type: {preferences.property_type.value}")
        print(f"  Bedrooms: {preferences.min_bedrooms}-{preferences.max_bedrooms}")
        print(f"  Price range: ${preferences.min_price}-${preferences.max_price}")
        print(f"  Amenities: {preferences.amenities}")
        print(f"  Lifestyle: {preferences.lifestyle_keywords}")
        print(f"  Neighborhoods: {preferences.neighborhoods}")


if __name__ == "__main__":
    main()
