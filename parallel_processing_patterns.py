#!/usr/bin/env python3
"""
Parallel Processing Patterns for Multicore Chips
Implements parallel processing strategies optimized for multicore architectures
like the Epiphany chip on Parallella boards.
"""

import threading
from typing import Callable, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from dataclasses import dataclass
import ctypes
import time
import os


@dataclass
class ParallelConfig:
    """Configuration for parallel processing"""
    num_workers: int
    chunk_size: int = 1024
    strategy: str = "threading"  # "threading", "multiprocessing", "custom"
    memory_limit: int = 1024 * 1024 * 256  # 256 MB default
    task_distribution: str = "round_robin"  # "round_robin", "work_stealing", "static_partition"


class ParallelProcessingPatterns:
    """Implementation of parallel processing patterns for multicore chips"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.num_workers = min(config.num_workers, os.cpu_count() or 1)
        
    def parallel_map(self, func: Callable, data: List[Any]) -> List[Any]:
        """Parallel map implementation using threading or multiprocessing"""
        if self.config.strategy == "threading":
            return self._threading_map(func, data)
        elif self.config.strategy == "multiprocessing":
            return self._multiprocessing_map(func, data)
        else:
            return self._custom_parallel_map(func, data)
    
    def _threading_map(self, func: Callable, data: List[Any]) -> List[Any]:
        """Thread-based parallel map"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(func, data))
        return results
    
    def _multiprocessing_map(self, func: Callable, data: List[Any]) -> List[Any]:
        """Process-based parallel map"""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(func, data))
        return results
    
    def _custom_parallel_map(self, func: Callable, data: List[Any]) -> List[Any]:
        """Custom parallel map with task distribution strategies"""
        if self.config.task_distribution == "round_robin":
            return self._round_robin_map(func, data)
        elif self.config.task_distribution == "work_stealing":
            return self._work_stealing_map(func, data)
        else:
            return self._static_partition_map(func, data)
    
    def _round_robin_map(self, func: Callable, data: List[Any]) -> List[Any]:
        """Round-robin task distribution"""
        if not data:
            return []
        
        # Split data into chunks
        chunks = [data[i:i + self.config.chunk_size] for i in range(0, len(data), self.config.chunk_size)]
        
        results = [None] * len(data)
        
        # Create worker threads
        threads = []
        for i in range(self.num_workers):
            thread_data = chunks[i::self.num_workers]  # Round-robin distribution
            flat_data = [item for chunk in thread_data for item in chunk]
            thread = threading.Thread(target=self._process_chunk_round_robin, 
                                    args=(func, flat_data, results, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return results
    
    def _process_chunk_round_robin(self, func: Callable, data_chunk: List[Any], 
                                 results: List[Any], worker_id: int):
        """Process a chunk of data in round-robin fashion"""
        for i, item in enumerate(data_chunk):
            result = func(item)
            # Calculate the correct index in the global results array
            global_idx = worker_id + i * self.num_workers
            if global_idx < len(results):
                results[global_idx] = result
    
    def _work_stealing_map(self, func: Callable, data: List[Any]) -> List[Any]:
        """Work-stealing task distribution pattern"""
        if not data:
            return []
        
        # Create a shared work queue
        work_queue = data.copy()
        result_lock = threading.Lock()
        results = [None] * len(data)
        completed_tasks = 0
        
        def worker():
            nonlocal completed_tasks
            while True:
                # Try to get work from the queue
                with threading.Lock():
                    if not work_queue:
                        break
                    item = work_queue.pop(0)
                    item_idx = data.index(item)
                
                # Process the item
                result = func(item)
                
                # Store result
                with result_lock:
                    results[item_idx] = result
                    completed_tasks += 1
        
        # Start worker threads
        threads = []
        for _ in range(self.num_workers):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return results
    
    def _static_partition_map(self, func: Callable, data: List[Any]) -> List[Any]:
        """Static partitioning of work"""
        if not data:
            return []
        
        # Partition data evenly among workers
        chunk_size = len(data) // self.num_workers
        if chunk_size == 0:
            chunk_size = 1
        
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        results = [None] * len(data)
        
        def process_chunk(chunk_data, start_idx):
            for i, item in enumerate(chunk_data):
                result = func(item)
                results[start_idx + i] = result
        
        # Process chunks in parallel
        threads = []
        for i, chunk in enumerate(chunks):
            start_idx = i * chunk_size
            thread = threading.Thread(target=process_chunk, args=(chunk, start_idx))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return results

    def parallel_reduce(self, func: Callable, data: List[Any], combiner: Callable) -> Any:
        """Parallel reduce implementation"""
        if len(data) <= 1:
            return data[0] if data else None
        
        # Partition data for parallel processing
        chunk_size = max(1, len(data) // self.num_workers)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Process each chunk in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            partial_results = list(executor.map(lambda chunk: self._sequential_reduce(func, chunk), chunks))
        
        # Combine partial results
        result = partial_results[0]
        for partial_result in partial_results[1:]:
            result = combiner(result, partial_result)
        
        return result
    
    def _sequential_reduce(self, func: Callable, chunk: List[Any]) -> Any:
        """Sequential reduce for a chunk"""
        if not chunk:
            return None
        
        result = chunk[0]
        for item in chunk[1:]:
            result = func(result, item)
        
        return result
    
    def parallel_pipeline(self, stages: List[Callable], data: List[Any]) -> List[Any]:
        """Parallel pipeline execution where different stages can run in parallel"""
        result = data
        for stage in stages:
            # Process all data through each stage
            result = self.parallel_map(stage, result)
        return result
    
    def data_parallel_pattern(self, func: Callable, data: List[Any]) -> List[Any]:
        """Data parallel pattern with automatic load balancing"""
        # For multicore chips like Epiphany, implement a simple parallel pattern
        # that distributes work evenly
        if len(data) <= self.num_workers:
            # Process directly if there are fewer data items than workers
            return [func(item) for item in data]
        
        # Partition the data
        partition_size = len(data) // self.num_workers
        partitions = [data[i:i + partition_size] for i in range(0, len(data), partition_size)]
        
        # Process each partition in parallel
        results = [None] * len(data)
        
        def process_partition(partition, start_idx):
            for i, item in enumerate(partition):
                result = func(item)
                results[start_idx + i] = result
        
        threads = []
        for i, partition in enumerate(partitions):
            start_idx = i * partition_size
            thread = threading.Thread(target=process_partition, args=(partition, start_idx))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return results


class EpiphanyOptimizedProcessor:
    """Optimized processor for Epiphany multicore architecture"""
    
    def __init__(self, num_cores: int = 16):
        self.num_cores = num_cores
        self.memory_per_core = 64 * 1024  # 64KB per core
        
    def distribute_work_epiphany_style(self, work_items: List[Any]) -> List[Tuple[int, List[Any]]]:
        """Distribute work in an Epiphany-optimized way (like row/column distribution)"""
        # Distribute work among "cores" in a grid pattern
        rows = int(self.num_cores ** 0.5)
        cols = self.num_cores // rows
        
        distribution = []
        for core_id in range(self.num_cores):
            # Assign work items to each "core" in a round-robin fashion
            core_work = work_items[core_id::self.num_cores]
            distribution.append((core_id, core_work))
        
        return distribution
    
    def simulate_epiphany_processing(self, func: Callable, data: List[Any]) -> List[Any]:
        """Simulate Epiphany-style processing with distributed work"""
        # Distribute work
        work_distribution = self.distribute_work_epiphany_style(data)
        
        # Process work on each "core" in parallel
        results = [None] * len(data)
        
        def process_core(core_id, core_data):
            for i, item in enumerate(core_data):
                result = func(item)
                # Calculate the global index for this result
                global_idx = core_id + i * self.num_cores
                if global_idx < len(results):
                    results[global_idx] = result
        
        # Execute all "cores" in parallel
        threads = []
        for core_id, core_data in work_distribution:
            thread = threading.Thread(target=process_core, args=(core_id, core_data))
            threads.append(thread)
            thread.start()
        
        # Wait for all "cores" to complete
        for thread in threads:
            thread.join()
        
        return results
    
    def epiphany_memory_optimized_map(self, func: Callable, data: List[Any]) -> List[Any]:
        """Memory-optimized mapping for Epiphany's constraints"""
        # Process data in chunks to respect memory constraints
        results = []
        
        chunk_size = self.memory_per_core // 8  # Rough estimate for 8 bytes per data element
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        for chunk in chunks:
            # Simulate processing on Epiphany cores
            chunk_results = self.simulate_epiphany_processing(func, chunk)
            results.extend(chunk_results)
        
        return results


def create_parallel_processing_demo():
    """Create a demonstration of parallel processing patterns"""
    
    # Example function to process (shader-like computation)
    def shader_operation(data):
        # Simulate some computation that might happen in a shader
        x, y = data
        result = (x * 0.5 + y * 0.3) % 1.0  # Simple color calculation
        return (result, result * 0.8, result * 0.6)
    
    # Create test data
    test_data = [(x, y) for x in range(0, 100) for y in range(0, 100)]
    
    # Test different parallel configurations
    configs = [
        ParallelConfig(num_workers=4, strategy="threading", task_distribution="round_robin"),
        ParallelConfig(num_workers=4, strategy="threading", task_distribution="work_stealing"),
        ParallelConfig(num_workers=4, strategy="threading", task_distribution="static_partition"),
    ]
    
    print("Testing different parallel processing patterns:")
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config.strategy}, {config.task_distribution}")
        
        processor = ParallelProcessingPatterns(config)
        
        start_time = time.time()
        results = processor.parallel_map(shader_operation, test_data[:1000])  # Limit for demo
        end_time = time.time()
        
        print(f"Processed {len(results)} items in {end_time - start_time:.4f} seconds")
        print(f"First few results: {results[:5]}")
    
    # Test Epiphany-optimized processor
    print("\nTesting Epiphany-optimized processor:")
    epiphany_processor = EpiphanyOptimizedProcessor(num_cores=16)
    
    start_time = time.time()
    epiphany_results = epiphany_processor.epiphany_memory_optimized_map(
        shader_operation, test_data[:1000]
    )
    end_time = time.time()
    
    print(f"Epiphany processed {len(epiphany_results)} items in {end_time - start_time:.4f} seconds")
    print(f"First few results: {epiphany_results[:5]}")


def optimize_for_multicore_shaders():
    """Function to demonstrate multicore shader optimization patterns"""
    print("Implementing parallel processing patterns for multicore chips...")
    
    # Create default configuration for Epiphany-like system
    epiphany_config = ParallelConfig(
        num_workers=16,  # 16 cores
        chunk_size=64,   # Small chunks for memory efficiency
        strategy="threading",  # Using threading as simulation
        memory_limit=32 * 1024 * 1024,  # 32MB total
        task_distribution="static_partition"
    )
    
    # Create a sample shader-like function that would benefit from parallelization
    def compute_pixel_shader(x, y, time):
        """Example pixel shader computation"""
        # Simulate some computation that's done per pixel
        r = (np.sin(x * 0.1 + time) + 1) * 0.5
        g = (np.cos(y * 0.1 + time) + 1) * 0.5
        b = (np.sin((x + y) * 0.05 + time) + 1) * 0.5
        return (r, g, b)
    
    # Generate pixel coordinates to process
    width, height = 64, 64  # Small test image
    pixels = [(x, y, 0.5) for y in range(height) for x in range(width)]
    
    # Process using parallel patterns
    processor = ParallelProcessingPatterns(epiphany_config)
    
    start_time = time.time()
    results = processor.data_parallel_pattern(
        lambda pixel: compute_pixel_shader(*pixel), 
        pixels
    )
    end_time = time.time()
    
    print(f"Processed {len(results)} pixels in {end_time - start_time:.4f} seconds")
    
    # Also test with Epiphany-optimized processor
    epiphany_processor = EpiphanyOptimizedProcessor(num_cores=16)
    
    start_time = time.time()
    epiphany_results = epiphany_processor.simulate_epiphany_processing(
        lambda pixel: compute_pixel_shader(*pixel),
        pixels
    )
    end_time = time.time()
    
    print(f"Epiphany-optimized processing: {len(epiphany_results)} pixels in {end_time - start_time:.4f} seconds")
    
    print("Parallel processing patterns for multicore chips implemented successfully!")


if __name__ == "__main__":
    optimize_for_multicore_shaders()