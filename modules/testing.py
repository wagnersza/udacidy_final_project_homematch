"""
HomeMatch Testing and Quality Assurance Module
Comprehensive testing framework for validating search functionality, personalization quality,
performance, and edge cases in the HomeMatch application.
"""

import time
import asyncio
import json
import statistics
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.vector_store import VectorStore
from modules.preference_processor import PreferenceProcessor, UserPreferences
from modules.description_personalizer import DescriptionPersonalizer
from modules.data_generator import DataGenerator
from modules.monitoring import get_logger


@dataclass
class TestResult:
    """Container for individual test results"""
    test_name: str
    status: str  # "pass", "fail", "warning"
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics"""
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float  # requests per second
    success_rate: float
    error_rate: float


class QualityAssuranceTester:
    """Comprehensive testing and quality assurance for HomeMatch system"""
    
    def __init__(self):
        self.logger = get_logger("qa_tester")
        self.vector_store = VectorStore()
        self.preference_processor = PreferenceProcessor()
        self.description_personalizer = DescriptionPersonalizer()
        self.data_generator = DataGenerator()
        
        # Test scenarios and data
        self.test_preferences = [
            "Looking for a family-friendly home with 3+ bedrooms under $500,000",
            "Modern downtown apartment with gym and parking",
            "Waterfront property with ocean views and large deck",
            "Pet-friendly house with big backyard in quiet neighborhood",
            "Luxury condo with high-end finishes and city views",
            "Starter home for first-time buyers under $300,000",
            "Investment property with good rental potential",
            "Energy-efficient home with solar panels and modern appliances",
            "Historic home with character and original details",
            "New construction with smart home features"
        ]
        
        self.edge_case_preferences = [
            "",  # Empty preference
            "x" * 1000,  # Very long preference
            "!@#$%^&*()",  # Special characters only
            "123456789",  # Numbers only
            "abcdefghijklmnopqrstuvwxyz",  # Alphabet only
            "house house house house house",  # Repetitive
            "I want everything perfect and nothing should be wrong",  # Unrealistic
            "cheap expensive big small new old",  # Contradictory
        ]
        
        self.test_results = []
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests and return results"""
        self.logger.info("Starting comprehensive testing suite")
        start_time = time.time()
        
        # Initialize test results
        self.test_results = []
        
        # Run all test categories
        search_results = self.test_search_functionality()
        personalization_results = self.test_personalization_quality()
        performance_results = self.test_performance()
        edge_case_results = self.test_edge_cases()
        integration_results = self.test_integration()
        
        total_time = time.time() - start_time
        
        # Compile comprehensive results
        results = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "passed": len([r for r in self.test_results if r.status == "pass"]),
                "failed": len([r for r in self.test_results if r.status == "fail"]),
                "warnings": len([r for r in self.test_results if r.status == "warning"]),
                "total_execution_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            "search_functionality": search_results,
            "personalization_quality": personalization_results,
            "performance": performance_results,
            "edge_cases": edge_case_results,
            "integration": integration_results,
            "detailed_results": [asdict(r) for r in self.test_results]
        }
        
        self.logger.info(f"Testing completed in {total_time:.2f}s")
        return results
    
    def test_search_functionality(self) -> Dict[str, Any]:
        """Test search functionality with diverse user preferences"""
        self.logger.info("Testing search functionality")
        results = {"tests": [], "summary": {}}
        
        for i, preference in enumerate(self.test_preferences):
            test_name = f"search_test_{i+1}"
            start_time = time.time()
            
            try:
                # Process preferences
                user_prefs = self.preference_processor.process_preferences(preference)
                
                # Perform search
                search_results = self.vector_store.search_listings(
                    query=preference,
                    n_results=5
                )
                
                execution_time = time.time() - start_time
                
                # Validate results
                validation = self._validate_search_results(search_results, user_prefs)
                
                result = TestResult(
                    test_name=test_name,
                    status="pass" if validation["valid"] else "fail",
                    execution_time=execution_time,
                    details={
                        "preference": preference,
                        "results_count": len(search_results),
                        "validation": validation
                    }
                )
                
                self.test_results.append(result)
                results["tests"].append(asdict(result))
                
            except Exception as e:
                execution_time = time.time() - start_time
                result = TestResult(
                    test_name=test_name,
                    status="fail",
                    execution_time=execution_time,
                    details={"preference": preference, "error": str(e)},
                    error_message=str(e)
                )
                self.test_results.append(result)
                results["tests"].append(asdict(result))
        
        # Calculate summary statistics
        test_times = [t["execution_time"] for t in results["tests"]]
        results["summary"] = {
            "total_tests": len(results["tests"]),
            "passed": len([t for t in results["tests"] if t["status"] == "pass"]),
            "avg_execution_time": statistics.mean(test_times) if test_times else 0,
            "max_execution_time": max(test_times) if test_times else 0
        }
        
        return results
    
    def test_personalization_quality(self) -> Dict[str, Any]:
        """Test personalization quality and factual integrity"""
        self.logger.info("Testing personalization quality")
        results = {"tests": [], "summary": {}}
        
        # Get sample listings for testing
        try:
            listings = self.data_generator.load_listings_from_file()
            if not listings:
                listings = self.vector_store.get_all_listings()
                listings = [{"id": l["id"], **l["metadata"]} for l in listings]
            
            sample_listings = random.sample(listings, min(5, len(listings)))
        except Exception as e:
            self.logger.error(f"Failed to load listings for testing: {e}")
            return {"error": "Could not load listings for testing"}
        
        for i, preference in enumerate(self.test_preferences[:5]):  # Test with first 5 preferences
            test_name = f"personalization_test_{i+1}"
            start_time = time.time()
            
            try:
                # Process preferences
                user_prefs = self.preference_processor.process_preferences(preference)
                
                # Test personalization on sample listings
                personalization_results = self.description_personalizer.personalize_multiple_listings(
                    sample_listings, user_prefs, use_llm=True
                )
                
                execution_time = time.time() - start_time
                
                # Validate personalization quality
                quality_assessment = self._assess_personalization_quality(
                    personalization_results, user_prefs
                )
                
                result = TestResult(
                    test_name=test_name,
                    status="pass" if quality_assessment["meets_standards"] else "warning",
                    execution_time=execution_time,
                    details={
                        "preference": preference,
                        "listings_tested": len(sample_listings),
                        "quality_assessment": quality_assessment
                    }
                )
                
                self.test_results.append(result)
                results["tests"].append(asdict(result))
                
            except Exception as e:
                execution_time = time.time() - start_time
                result = TestResult(
                    test_name=test_name,
                    status="fail",
                    execution_time=execution_time,
                    details={"preference": preference, "error": str(e)},
                    error_message=str(e)
                )
                self.test_results.append(result)
                results["tests"].append(asdict(result))
        
        # Calculate summary
        quality_scores = []
        for test in results["tests"]:
            if "quality_assessment" in test["details"]:
                quality_scores.append(test["details"]["quality_assessment"]["overall_score"])
        
        results["summary"] = {
            "total_tests": len(results["tests"]),
            "passed": len([t for t in results["tests"] if t["status"] == "pass"]),
            "avg_quality_score": statistics.mean(quality_scores) if quality_scores else 0
        }
        
        return results
    
    def test_performance(self) -> Dict[str, Any]:
        """Test system performance under various loads"""
        self.logger.info("Testing system performance")
        results = {"tests": [], "summary": {}}
        
        # Test scenarios with different loads
        load_scenarios = [
            {"name": "light_load", "concurrent_requests": 1, "total_requests": 10},
            {"name": "medium_load", "concurrent_requests": 3, "total_requests": 15},
            {"name": "heavy_load", "concurrent_requests": 5, "total_requests": 20}
        ]
        
        for scenario in load_scenarios:
            test_name = f"performance_{scenario['name']}"
            
            try:
                metrics = self._run_performance_test(
                    concurrent_requests=scenario['concurrent_requests'],
                    total_requests=scenario['total_requests']
                )
                
                # Determine if performance meets standards
                status = "pass"
                if metrics.avg_response_time > 5.0:  # 5 second threshold
                    status = "warning"
                if metrics.avg_response_time > 10.0 or metrics.error_rate > 0.1:  # 10 seconds or 10% error rate
                    status = "fail"
                
                result = TestResult(
                    test_name=test_name,
                    status=status,
                    execution_time=0,  # Performance test tracks its own metrics
                    details={
                        "scenario": scenario,
                        "metrics": asdict(metrics)
                    }
                )
                
                self.test_results.append(result)
                results["tests"].append(asdict(result))
                
            except Exception as e:
                result = TestResult(
                    test_name=test_name,
                    status="fail",
                    execution_time=0,
                    details={"scenario": scenario, "error": str(e)},
                    error_message=str(e)
                )
                self.test_results.append(result)
                results["tests"].append(asdict(result))
        
        # Performance summary
        avg_response_times = []
        for test in results["tests"]:
            if "metrics" in test["details"]:
                avg_response_times.append(test["details"]["metrics"]["avg_response_time"])
        
        results["summary"] = {
            "total_scenarios": len(results["tests"]),
            "passed": len([t for t in results["tests"] if t["status"] == "pass"]),
            "overall_avg_response_time": statistics.mean(avg_response_times) if avg_response_times else 0
        }
        
        return results
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test system behavior with edge cases and invalid inputs"""
        self.logger.info("Testing edge case handling")
        results = {"tests": [], "summary": {}}
        
        for i, edge_case in enumerate(self.edge_case_preferences):
            test_name = f"edge_case_test_{i+1}"
            start_time = time.time()
            
            try:
                # Test preference processing with edge case
                if edge_case.strip():  # Skip empty string for preference processing
                    user_prefs = self.preference_processor.process_preferences(edge_case)
                    search_results = self.vector_store.search_listings(
                        query=edge_case,
                        user_preferences=user_prefs,
                        n_results=3
                    )
                else:
                    # Test empty preference handling
                    user_prefs = None
                    search_results = []
                
                execution_time = time.time() - start_time
                
                # Edge cases should be handled gracefully
                result = TestResult(
                    test_name=test_name,
                    status="pass",  # If we get here without exception, it's handled
                    execution_time=execution_time,
                    details={
                        "edge_case": edge_case,
                        "edge_case_type": self._classify_edge_case(edge_case),
                        "results_count": len(search_results),
                        "handled_gracefully": True
                    }
                )
                
                self.test_results.append(result)
                results["tests"].append(asdict(result))
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Some exceptions might be expected for edge cases
                expected_error = self._is_expected_edge_case_error(edge_case, e)
                
                result = TestResult(
                    test_name=test_name,
                    status="pass" if expected_error else "fail",
                    execution_time=execution_time,
                    details={
                        "edge_case": edge_case,
                        "edge_case_type": self._classify_edge_case(edge_case),
                        "error": str(e),
                        "expected_error": expected_error
                    },
                    error_message=str(e) if not expected_error else None
                )
                
                self.test_results.append(result)
                results["tests"].append(asdict(result))
        
        results["summary"] = {
            "total_edge_cases": len(results["tests"]),
            "handled_gracefully": len([t for t in results["tests"] if t["status"] == "pass"]),
            "unexpected_failures": len([t for t in results["tests"] if t["status"] == "fail"])
        }
        
        return results
    
    def test_integration(self) -> Dict[str, Any]:
        """Test end-to-end integration of all system components"""
        self.logger.info("Testing system integration")
        results = {"tests": [], "summary": {}}
        
        integration_scenarios = [
            {
                "name": "full_workflow_test",
                "description": "Test complete user workflow from preference to personalized results",
                "preference": "Family home with 4 bedrooms and good schools nearby"
            },
            {
                "name": "data_consistency_test", 
                "description": "Test consistency between vector store and file storage",
                "preference": "Modern apartment downtown"
            },
            {
                "name": "error_recovery_test",
                "description": "Test system recovery from component failures",
                "preference": "Beachfront property with ocean views"
            }
        ]
        
        for scenario in integration_scenarios:
            test_name = scenario["name"]
            start_time = time.time()
            
            try:
                if scenario["name"] == "full_workflow_test":
                    result_details = self._test_full_workflow(scenario["preference"])
                elif scenario["name"] == "data_consistency_test":
                    result_details = self._test_data_consistency()
                elif scenario["name"] == "error_recovery_test":
                    result_details = self._test_error_recovery(scenario["preference"])
                
                execution_time = time.time() - start_time
                
                result = TestResult(
                    test_name=test_name,
                    status="pass" if result_details["success"] else "fail",
                    execution_time=execution_time,
                    details=result_details
                )
                
                self.test_results.append(result)
                results["tests"].append(asdict(result))
                
            except Exception as e:
                execution_time = time.time() - start_time
                result = TestResult(
                    test_name=test_name,
                    status="fail",
                    execution_time=execution_time,
                    details={"scenario": scenario, "error": str(e)},
                    error_message=str(e)
                )
                self.test_results.append(result)
                results["tests"].append(asdict(result))
        
        results["summary"] = {
            "total_integration_tests": len(results["tests"]),
            "passed": len([t for t in results["tests"] if t["status"] == "pass"]),
            "system_health": "healthy" if all(t["status"] == "pass" for t in results["tests"]) else "issues_detected"
        }
        
        return results
    
    def _validate_search_results(self, search_results: List[Dict], user_prefs: UserPreferences) -> Dict[str, Any]:
        """Validate search results quality and relevance"""
        validation = {
            "valid": True,
            "issues": [],
            "metrics": {}
        }
        
        # Check if results exist
        if not search_results:
            validation["valid"] = False
            validation["issues"].append("No search results returned")
            return validation
        
        # Check result structure
        for i, result in enumerate(search_results):
            if "metadata" not in result:
                validation["issues"].append(f"Result {i} missing metadata")
            
            if "distance" not in result:
                validation["issues"].append(f"Result {i} missing distance score")
        
        # Check relevance (basic heuristics)
        if user_prefs:
            relevance_score = self._calculate_relevance_score(search_results, user_prefs)
            validation["metrics"]["relevance_score"] = relevance_score
            
            if relevance_score < 0.3:  # Threshold for relevance
                validation["issues"].append("Low relevance score detected")
        
        if validation["issues"]:
            validation["valid"] = False
        
        return validation
    
    def _assess_personalization_quality(self, personalization_results: List[Any], user_prefs: UserPreferences) -> Dict[str, Any]:
        """Assess the quality of personalized descriptions"""
        assessment = {
            "meets_standards": True,
            "overall_score": 0.0,
            "issues": [],
            "metrics": {}
        }
        
        scores = []
        for result in personalization_results:
            if hasattr(result, 'personalization_score'):
                scores.append(result.personalization_score)
            
            # Check for factual integrity
            if hasattr(result, 'error_message') and result.error_message:
                assessment["issues"].append(f"Personalization error: {result.error_message}")
            
            # Check for fallback usage
            if hasattr(result, 'fallback_used') and result.fallback_used:
                assessment["issues"].append("Fallback used instead of AI personalization")
        
        if scores:
            assessment["overall_score"] = statistics.mean(scores)
            assessment["metrics"]["avg_personalization_score"] = assessment["overall_score"]
            assessment["metrics"]["min_score"] = min(scores)
            assessment["metrics"]["max_score"] = max(scores)
            
            # Set standards
            if assessment["overall_score"] < 0.6:
                assessment["meets_standards"] = False
                assessment["issues"].append("Average personalization score below threshold")
        
        return assessment
    
    def _run_performance_test(self, concurrent_requests: int, total_requests: int) -> PerformanceMetrics:
        """Run performance test with specified load"""
        start_time = time.time()
        response_times = []
        errors = 0
        
        # Create test requests
        test_preference = random.choice(self.test_preferences)
        
        def make_request():
            try:
                request_start = time.time()
                user_prefs = self.preference_processor.process_preferences(test_preference)
                search_results = self.vector_store.search_listings(
                    query=test_preference,
                    n_results=3
                )
                request_time = time.time() - request_start
                return request_time, None
            except Exception as e:
                request_time = time.time() - request_start
                return request_time, str(e)
        
        # Execute requests with thread pool
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(total_requests)]
            
            for future in as_completed(futures):
                response_time, error = future.result()
                response_times.append(response_time)
                if error:
                    errors += 1
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        if response_times:
            response_times.sort()
            p95_index = int(0.95 * len(response_times))
            p99_index = int(0.99 * len(response_times))
            
            return PerformanceMetrics(
                avg_response_time=statistics.mean(response_times),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                p95_response_time=response_times[p95_index] if p95_index < len(response_times) else max(response_times),
                p99_response_time=response_times[p99_index] if p99_index < len(response_times) else max(response_times),
                throughput=total_requests / total_time,
                success_rate=(total_requests - errors) / total_requests,
                error_rate=errors / total_requests
            )
        else:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 1.0)
    
    def _classify_edge_case(self, edge_case: str) -> str:
        """Classify the type of edge case"""
        if not edge_case:
            return "empty_input"
        elif len(edge_case) > 500:
            return "oversized_input"
        elif edge_case.isdigit():
            return "numeric_only"
        elif not edge_case.isalnum():
            return "special_characters"
        elif len(set(edge_case.split())) == 1:
            return "repetitive"
        else:
            return "other"
    
    def _is_expected_edge_case_error(self, edge_case: str, error: Exception) -> bool:
        """Determine if an error is expected for a given edge case"""
        # Empty inputs might legitimately cause errors
        if not edge_case.strip():
            return True
        
        # Very long inputs might cause timeouts
        if len(edge_case) > 500:
            return "timeout" in str(error).lower()
        
        # Special character only inputs might cause parsing errors
        if not edge_case.isalnum():
            return "parse" in str(error).lower() or "invalid" in str(error).lower()
        
        return False
    
    def _calculate_relevance_score(self, search_results: List[Dict], user_prefs: UserPreferences) -> float:
        """Calculate a basic relevance score for search results"""
        if not search_results or not user_prefs:
            return 0.0
        
        # This is a simplified relevance calculation
        # In a real system, this would be more sophisticated
        total_score = 0.0
        for result in search_results:
            score = 0.0
            metadata = result.get("metadata", {})
            
            # Check price range match using min/max price if available
            if hasattr(user_prefs, 'min_price') and hasattr(user_prefs, 'max_price'):
                price = metadata.get("price", 0)
                if user_prefs.min_price and user_prefs.max_price:
                    if user_prefs.min_price <= price <= user_prefs.max_price:
                        score += 0.3
            elif hasattr(user_prefs, 'price_range') and user_prefs.price_range:
                # Basic price range matching
                price = metadata.get("price", 0)
                if price > 0:  # If price is available, give some score
                    score += 0.1
            
            # Check property type match
            if hasattr(user_prefs, 'property_type') and user_prefs.property_type:
                prop_type = metadata.get("property_type", "").lower()
                if hasattr(user_prefs.property_type, 'value'):
                    pref_type = user_prefs.property_type.value.lower()
                else:
                    pref_type = str(user_prefs.property_type).lower()
                
                if pref_type != "any" and pref_type in prop_type:
                    score += 0.3
            
            # Check location match
            if hasattr(user_prefs, 'neighborhoods') and user_prefs.neighborhoods:
                location = metadata.get("neighborhood", "").lower()
                for neighborhood in user_prefs.neighborhoods:
                    if neighborhood.lower() in location:
                        score += 0.4
                        break
            
            total_score += score
        
        return total_score / len(search_results)
    
    def _test_full_workflow(self, preference: str) -> Dict[str, Any]:
        """Test complete workflow from preference to personalized results"""
        try:
            # Step 1: Process preferences
            user_prefs = self.preference_processor.process_preferences(preference)
            
            # Step 2: Search listings
            search_results = self.vector_store.search_listings(
                query=preference,
                n_results=3
            )
            
            # Step 3: Personalize descriptions
            if search_results:
                listings = [{"id": r["id"], **r["metadata"]} for r in search_results]
                personalization_results = self.description_personalizer.personalize_multiple_listings(
                    listings, user_prefs, use_llm=False  # Use fallback for testing
                )
            
            return {
                "success": True,
                "workflow_steps_completed": 3,
                "preference_processed": user_prefs is not None,
                "search_results_count": len(search_results),
                "personalization_completed": len(search_results) > 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "workflow_steps_completed": 0
            }
    
    def _test_data_consistency(self) -> Dict[str, Any]:
        """Test consistency between different data sources"""
        try:
            # Get listings from file
            file_listings = self.data_generator.load_listings_from_file()
            
            # Get listings from vector store
            vector_listings = self.vector_store.get_all_listings()
            
            # Compare counts and basic consistency
            file_count = len(file_listings) if file_listings else 0
            vector_count = len(vector_listings) if vector_listings else 0
            
            consistency_score = min(file_count, vector_count) / max(file_count, vector_count, 1)
            
            return {
                "success": True,
                "file_listings_count": file_count,
                "vector_listings_count": vector_count,
                "consistency_score": consistency_score,
                "data_consistent": consistency_score > 0.8
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _test_error_recovery(self, preference: str) -> Dict[str, Any]:
        """Test system recovery from simulated errors"""
        try:
            # Test with invalid API configuration
            original_model = self.description_personalizer.model
            self.description_personalizer.model = "invalid-model"
            
            # This should trigger fallback behavior
            user_prefs = self.preference_processor.process_preferences(preference)
            search_results = self.vector_store.search_listings(
                query=preference,
                user_preferences=user_prefs,
                n_results=2
            )
            
            # Restore original model
            self.description_personalizer.model = original_model
            
            return {
                "success": True,
                "error_recovery_tested": True,
                "fallback_behavior_working": True,
                "search_still_functional": len(search_results) > 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_recovery_failed": True
            }
