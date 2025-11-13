# Performance Profiler for SuperShader

import time
import functools
from typing import Dict, List, Callable
import json

class PerformanceProfiler:
    def __init__(self):
        self.profiles = {}
        self.call_counts = {}
    
    def profile(self, name: str = None):
        """Decorator to profile a function."""
        def decorator(func: Callable):
            func_name = name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = None
                    success = False
                    print(f"Error in {func_name}: {str(e)}")
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Update profiles
                if func_name not in self.profiles:
                    self.profiles[func_name] = {
                        "total_time": 0,
                        "call_count": 0,
                        "min_time": float("inf"),
                        "max_time": 0,
                        "errors": 0
                    }
                
                profile = self.profiles[func_name]
                profile["total_time"] += execution_time
                profile["call_count"] += 1
                profile["min_time"] = min(profile["min_time"], execution_time)
                profile["max_time"] = max(profile["max_time"], execution_time)
                if not success:
                    profile["errors"] += 1
                
                return result
            return wrapper
        return decorator
    
    def get_report(self) -> str:
        """Generate a performance report."""
        if not self.profiles:
            return "No profiling data available."
        
        report = ["Performance Report", "=" * 20]
        
        # Sort by total time
        sorted_profiles = sorted(
            self.profiles.items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )
        
        for func_name, stats in sorted_profiles:
            avg_time = stats["total_time"] / stats["call_count"]
            report.append(f"\n{func_name}:")
            report.append(f"  Calls: {stats['call_count']}")
            report.append(f"  Total Time: {stats['total_time']:.4f}s")
            report.append(f"  Average Time: {avg_time:.4f}s")
            report.append(f"  Min Time: {stats['min_time']:.4f}s")
            report.append(f"  Max Time: {stats['max_time']:.4f}s")
            report.append(f"  Errors: {stats['errors']}")
        
        return "\n".join(report)
    
    def save_report(self, filename: str):
        """Save profile data to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.profiles, f, indent=2)
    
    def reset(self):
        """Reset all profiling data."""
        self.profiles = {}
        self.call_counts = {}

# Global profiler instance
profiler = PerformanceProfiler()
