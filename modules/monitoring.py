"""
HomeMatch Logging and Monitoring Module
Comprehensive logging, monitoring, and metrics collection for the HomeMatch application
"""

import time
import logging
import functools
import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import json
from pathlib import Path


class MetricsCollector:
    """Collect and manage application metrics"""
    
    def __init__(self):
        self._metrics = defaultdict(lambda: defaultdict(int))
        self._timing_metrics = defaultdict(list)
        self._error_metrics = defaultdict(int)
        self._request_history = deque(maxlen=1000)  # Keep last 1000 requests
        self._lock = threading.Lock()
        self._start_time = datetime.now()
    
    def increment_counter(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self._lock:
            key = self._build_metric_key(metric_name, tags)
            self._metrics["counters"][key] += 1
    
    def record_timing(self, metric_name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        with self._lock:
            key = self._build_metric_key(metric_name, tags)
            self._timing_metrics[key].append(duration)
            # Keep only last 100 measurements per metric
            if len(self._timing_metrics[key]) > 100:
                self._timing_metrics[key] = self._timing_metrics[key][-100:]
    
    def record_error(self, error_type: str, tags: Optional[Dict[str, str]] = None):
        """Record an error metric"""
        with self._lock:
            key = self._build_metric_key(error_type, tags)
            self._error_metrics[key] += 1
    
    def record_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record a request for monitoring"""
        with self._lock:
            request_data = {
                "timestamp": datetime.now().isoformat(),
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "duration": duration
            }
            self._request_history.append(request_data)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        with self._lock:
            summary = {
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
                "counters": dict(self._metrics["counters"]),
                "errors": dict(self._error_metrics),
                "timing_stats": {},
                "request_stats": self._get_request_stats()
            }
            
            # Calculate timing statistics
            for metric, timings in self._timing_metrics.items():
                if timings:
                    summary["timing_stats"][metric] = {
                        "count": len(timings),
                        "avg": sum(timings) / len(timings),
                        "min": min(timings),
                        "max": max(timings),
                        "p95": self._percentile(timings, 95),
                        "p99": self._percentile(timings, 99)
                    }
            
            return summary
    
    def _build_metric_key(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Build a metric key with tags"""
        if not tags:
            return metric_name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric_name}[{tag_str}]"
    
    def _percentile(self, data: list, percentile: int) -> float:
        """Calculate percentile of a list"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile / 100
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    def _get_request_stats(self) -> Dict[str, Any]:
        """Get request statistics from history"""
        if not self._request_history:
            return {}
        
        recent_requests = list(self._request_history)
        total_requests = len(recent_requests)
        
        # Count by status code
        status_counts = defaultdict(int)
        endpoint_counts = defaultdict(int)
        total_duration = 0
        
        for req in recent_requests:
            status_counts[req["status_code"]] += 1
            endpoint_counts[req["endpoint"]] += 1
            total_duration += req["duration"]
        
        return {
            "total_requests": total_requests,
            "avg_response_time": total_duration / total_requests if total_requests > 0 else 0,
            "status_codes": dict(status_counts),
            "popular_endpoints": dict(sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }


class HomeMatchLogger:
    """Enhanced logger for HomeMatch application"""
    
    def __init__(self, name: str = "homewatch", log_config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.metrics = MetricsCollector()
        
        if log_config:
            logging.config.dictConfig(log_config)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message with optional extra data"""
        self._log_with_metrics("info", message, extra, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log error message with optional exception and extra data"""
        if error:
            self.metrics.record_error(type(error).__name__)
            extra = extra or {}
            extra.update({"error_type": type(error).__name__, "error_message": str(error)})
        self._log_with_metrics("error", message, extra, **kwargs)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message with optional extra data"""
        self._log_with_metrics("warning", message, extra, **kwargs)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message with optional extra data"""
        self._log_with_metrics("debug", message, extra, **kwargs)
    
    def _log_with_metrics(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Internal method to log with metrics collection"""
        # Increment log level counter
        self.metrics.increment_counter(f"log_{level}")
        
        # Add extra context
        log_extra = extra or {}
        log_extra.update(kwargs)
        
        # Log the message
        getattr(self.logger, level)(message, extra=log_extra)


def timing_decorator(metric_name: str, logger: HomeMatchLogger):
    """Decorator to measure execution time of functions"""
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.metrics.record_timing(metric_name, duration)
                    logger.debug(f"{func.__name__} completed in {duration:.3f}s")
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.metrics.record_timing(f"{metric_name}_error", duration)
                    logger.error(f"{func.__name__} failed after {duration:.3f}s", error=e)
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.metrics.record_timing(metric_name, duration)
                    logger.debug(f"{func.__name__} completed in {duration:.3f}s")
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.metrics.record_timing(f"{metric_name}_error", duration)
                    logger.error(f"{func.__name__} failed after {duration:.3f}s", error=e)
                    raise
            return sync_wrapper
    return decorator


def error_handler_decorator(logger: HomeMatchLogger):
    """Decorator to handle and log errors gracefully"""
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Unhandled error in {func.__name__}", error=e)
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Unhandled error in {func.__name__}", error=e)
                    raise
            return sync_wrapper
    return decorator


class HealthChecker:
    """Comprehensive health checking for application components"""
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = {}
        self.check_cache = {}
        self.cache_duration = 30  # Cache health checks for 30 seconds
    
    def register_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a health check function"""
        self.checks[name] = check_func
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check"""
        if name not in self.checks:
            return {"status": "error", "message": f"Health check '{name}' not found"}
        
        # Check cache
        now = time.time()
        if (name in self.last_check_time and 
            now - self.last_check_time[name] < self.cache_duration and
            name in self.check_cache):
            return self.check_cache[name]
        
        try:
            result = self.checks[name]()
            result["last_checked"] = datetime.now().isoformat()
            self.check_cache[name] = result
            self.last_check_time[name] = now
            return result
        except Exception as e:
            error_result = {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "last_checked": datetime.now().isoformat()
            }
            self.check_cache[name] = error_result
            self.last_check_time[name] = now
            return error_result
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {}
        overall_status = "healthy"
        
        for name in self.checks:
            result = self.run_check(name)
            results[name] = result
            
            if result.get("status") in ["error", "unhealthy"]:
                overall_status = "unhealthy"
            elif result.get("status") == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": results
        }


# Global instances
_logger = None
_health_checker = None


def get_logger(name: str = "homewatch") -> HomeMatchLogger:
    """Get or create the global logger instance"""
    global _logger
    if _logger is None:
        _logger = HomeMatchLogger(name)
    return _logger


def get_health_checker() -> HealthChecker:
    """Get or create the global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def setup_logging(log_config: Dict[str, Any]):
    """Setup logging configuration"""
    global _logger
    _logger = HomeMatchLogger(log_config=log_config)
    return _logger
