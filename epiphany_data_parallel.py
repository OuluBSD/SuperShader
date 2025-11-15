#!/usr/bin/env python3
"""
Data Parallel Processing Patterns for Epiphany Architecture
Implements efficient parallel processing patterns optimized for Epiphany's distributed architecture
"""

import numpy as np
from typing import List, Callable, Any, Dict, Tuple
from dataclasses import dataclass
import threading


@dataclass
class EpiphanyDataParallelConfig:
    """Configuration for Epiphany data parallel processing"""
    num_cores: int = 16
    rows: int = 4
    cols: int = 4
    chunk_size: int = 1024
    use_dma: bool = True
    memory_per_core: int = 32 * 1024  # 32KB per core
    shared_memory_size: int = 64 * 1024  # 64KB shared memory


class EpiphanyDataParallelProcessor:
    """Data parallel processor optimized for Epiphany architecture"""
    
    def __init__(self, config: EpiphanyDataParallelConfig = None):
        self.config = config or EpiphanyDataParallelConfig()
        self.num_cores = self.config.num_cores
        self.rows = self.config.rows
        self.cols = self.config.cols
    
    def parallel_map(self, func: Callable, data: List[Any]) -> List[Any]:
        """Parallel map optimized for Epiphany's architecture"""
        if len(data) == 0:
            return []
        
        # Calculate chunk size per core
        chunk_size = max(1, len(data) // self.num_cores)
        
        # Distribute work among cores (simulated)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Process chunks in parallel
        results = [None] * len(chunks)
        threads = []
        
        for i, chunk in enumerate(chunks):
            if i < self.num_cores:  # Only use as many threads as cores
                thread = threading.Thread(
                    target=self._process_chunk, 
                    args=(func, chunk, results, i)
                )
                threads.append(thread)
                thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Flatten results
        final_results = []
        for chunk_result in results:
            if chunk_result:
                final_results.extend(chunk_result)
        
        # Truncate to original data length (in case of padding)
        return final_results[:len(data)]
    
    def _process_chunk(self, func: Callable, chunk: List[Any], results: List[Any], core_id: int):
        """Process a chunk on a specific core (simulated)"""
        chunk_result = []
        for item in chunk:
            result = func(item)
            chunk_result.append(result)
        results[core_id] = chunk_result
    
    def parallel_reduce(self, func: Callable, data: List[Any], combiner: Callable) -> Any:
        """Parallel reduce operation optimized for Epiphany"""
        if len(data) <= 1:
            return data[0] if data else None
        
        # Split data among cores
        chunk_size = max(1, len(data) // self.num_cores)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Process each chunk in parallel
        partial_results = [None] * len(chunks)
        threads = []
        
        for i, chunk in enumerate(chunks):
            thread = threading.Thread(
                target=self._reduce_chunk,
                args=(func, chunk, partial_results, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for threads to complete
        for thread in threads:
            thread.join()
        
        # Combine partial results
        result = partial_results[0]
        for partial_result in partial_results[1:]:
            if partial_result is not None:
                result = combiner(result, partial_result)
        
        return result
    
    def _reduce_chunk(self, func: Callable, chunk: List[Any], results: List[Any], core_id: int):
        """Reduce a chunk on a specific core"""
        if not chunk:
            results[core_id] = None
            return
        
        result = chunk[0]
        for item in chunk[1:]:
            result = func(result, item)
        results[core_id] = result
    
    def scatter_gather_pattern(self, data: List[Any], process_func: Callable) -> List[Any]:
        """Implement scatter-gather pattern for Epiphany"""
        # Scatter: distribute data to cores
        scattered_data = self._scatter_data(data)
        
        # Process: each core processes its assigned data
        processed_data = [None] * len(scattered_data)
        threads = []
        
        for i, core_data in enumerate(scattered_data):
            thread = threading.Thread(
                target=self._process_scattered_data,
                args=(process_func, core_data, processed_data, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all cores to finish
        for thread in threads:
            thread.join()
        
        # Gather: collect results from all cores
        result = self._gather_data(processed_data)
        return result
    
    def _scatter_data(self, data: List[Any]) -> List[List[Any]]:
        """Scatter data among cores"""
        # Distribute data evenly among cores
        core_data = [[] for _ in range(self.num_cores)]
        
        for i, item in enumerate(data):
            core_id = i % self.num_cores
            core_data[core_id].append(item)
        
        return core_data
    
    def _process_scattered_data(self, func: Callable, core_data: List[Any], 
                               results: List[List[Any]], core_id: int):
        """Process data assigned to a core"""
        processed = []
        for item in core_data:
            processed.append(func(item))
        results[core_id] = processed
    
    def _gather_data(self, processed_data: List[List[Any]]) -> List[Any]:
        """Gather data from all cores"""
        # Interleave results from each core
        result = []
        max_len = max(len(core_data) for core_data in processed_data if core_data) if processed_data else 0
        
        for i in range(max_len):
            for core_data in processed_data:
                if i < len(core_data):
                    result.append(core_data[i])
        
        return result
    
    def map_reduce_pattern(self, data: List[Any], 
                          map_func: Callable, 
                          reduce_func: Callable) -> Any:
        """Map-reduce pattern optimized for Epiphany"""
        # Map phase: distribute and map data
        mapped_chunks = [None] * self.num_cores
        threads = []
        
        # Distribute data among cores
        for core_id in range(self.num_cores):
            start_idx = core_id * (len(data) // self.num_cores)
            end_idx = (core_id + 1) * (len(data) // self.num_cores) if core_id < self.num_cores - 1 else len(data)
            core_data = data[start_idx:end_idx]
            
            thread = threading.Thread(
                target=self._map_data,
                args=(map_func, core_data, mapped_chunks, core_id)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for mapping to complete
        for thread in threads:
            thread.join()
        
        # Reduce phase: combine results from each core
        all_mapped = []
        for chunk in mapped_chunks:
            if chunk:
                all_mapped.extend(chunk)
        
        # Do final reduction
        if not all_mapped:
            return None
        
        result = all_mapped[0]
        for item in all_mapped[1:]:
            result = reduce_func(result, item)
        
        return result
    
    def _map_data(self, map_func: Callable, data: List[Any], 
                 results: List[List[Any]], core_id: int):
        """Map data on a specific core"""
        mapped = [map_func(item) for item in data]
        results[core_id] = mapped


class EpiphanyDistributedMemoryProcessor:
    """Processor that takes advantage of Epiphany's distributed memory model"""
    
    def __init__(self, config: EpiphanyDataParallelConfig = None):
        self.config = config or EpiphanyDataParallelConfig()
        self.memory_per_core = self.config.memory_per_core
        
    def distribute_data_by_memory(self, data: List[Any], item_size_estimate: int = 1024) -> List[List[Any]]:
        """Distribute data based on memory constraints per core"""
        # Calculate how many items per core based on memory constraints
        items_per_core = self.memory_per_core // item_size_estimate
        items_per_core = max(1, items_per_core)  # Ensure at least 1 item per core
        
        distributed = [[] for _ in range(self.config.num_cores)]
        
        for i, item in enumerate(data):
            core_id = i % self.config.num_cores
            # Check if this core has space for more items
            if len(distributed[core_id]) < items_per_core:
                distributed[core_id].append(item)
            else:
                # Rotate to next core
                next_core = (core_id + 1) % self.config.num_cores
                distributed[next_core].append(item)
        
        return distributed
    
    def process_with_memory_optimization(self, data: List[Any], process_func: Callable) -> List[Any]:
        """Process data with memory optimization for Epiphany"""
        # Distribute data according to memory constraints
        distributed_data = self.distribute_data_by_memory(data)
        
        results = [None] * len(distributed_data)
        threads = []
        
        for i, core_data in enumerate(distributed_data):
            if core_data:  # Only create thread if core has data
                thread = threading.Thread(
                    target=self._process_with_memory_limit,
                    args=(process_func, core_data, results, i)
                )
                threads.append(thread)
                thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Reassemble results in original order
        final_result = []
        for core_result in results:
            if core_result:
                final_result.extend(core_result)
        
        return final_result
    
    def _process_with_memory_limit(self, func: Callable, data: List[Any], 
                                  results: List[List[Any]], core_id: int):
        """Process data with memory limitations"""
        # Simulate memory-limited processing
        result = []
        for item in data:
            processed_item = func(item)
            result.append(processed_item)
        results[core_id] = result


class EpiphanyParallelPipeline:
    """Complete parallel processing pipeline for Epiphany"""
    
    def __init__(self, config: EpiphanyDataParallelConfig = None):
        self.data_processor = EpiphanyDataParallelProcessor(config)
        self.memory_processor = EpiphanyDistributedMemoryProcessor(config)
        self.config = config or EpiphanyDataParallelConfig()
    
    def execute_pipeline(self, data: List[Any], stages: List[Callable]) -> List[Any]:
        """Execute a multi-stage pipeline in parallel across Epiphany cores"""
        result = data
        
        for i, stage in enumerate(stages):
            print(f"Executing pipeline stage {i+1}/{len(stages)}")
            
            # Use different parallelization strategy for each stage
            if i % 2 == 0:
                # Use map-reduce for even stages
                result = self.data_processor.map_reduce_pattern(
                    result, 
                    stage, 
                    lambda x, y: x + y if isinstance(x, (list, tuple)) else (x if x > y else y)
                )
                # Convert result back to list format if needed for next stage
                if not isinstance(result, list):
                    result = [result] * len(data)
            else:
                # Use simple parallel map for odd stages
                result = self.data_processor.parallel_map(stage, result)
        
        return result


def create_data_parallel_patterns():
    """Create and demonstrate data parallel processing patterns for Epiphany"""
    print("Creating data parallel processing patterns for Epiphany architecture...")
    
    # Create configuration for Epiphany
    config = EpiphanyDataParallelConfig(num_cores=16, memory_per_core=32*1024)
    processor = EpiphanyDataParallelProcessor(config)
    
    # Example functions to use with parallel patterns
    def pixel_shader_func(pixel_data):
        """Example pixel processing function"""
        x, y, time = pixel_data
        r = (np.sin(x * 0.01 + time) + 1) * 0.5
        g = (np.cos(y * 0.01 + time) + 1) * 0.5
        b = (np.sin((x + y) * 0.005 + time) + 1) * 0.5
        return (r, g, b, 1.0)
    
    # Create test data
    width, height = 100, 100
    time_val = 0.5
    test_data = [(x, y, time_val) for y in range(height) for x in range(width)][:500]  # Limit for demo
    
    print(f"Processing {len(test_data)} data items using parallel patterns...")
    
    # Test parallel map
    start_time = __import__('time').time()
    mapped_result = processor.parallel_map(pixel_shader_func, test_data)
    end_time = __import__('time').time()
    print(f"Parallel map completed in {end_time - start_time:.4f} seconds")
    print(f"First few results: {mapped_result[:3]}")
    
    # Test scatter-gather pattern
    start_time = __import__('time').time()
    scatter_gather_result = processor.scatter_gather_pattern(test_data, pixel_shader_func)
    end_time = __import__('time').time()
    print(f"Scatter-gather completed in {end_time - start_time:.4f} seconds")
    
    # Test with memory-optimized processor
    memory_processor = EpiphanyDistributedMemoryProcessor(config)
    start_time = __import__('time').time()
    memory_opt_result = memory_processor.process_with_memory_optimization(
        test_data, pixel_shader_func
    )
    end_time = __import__('time').time()
    print(f"Memory-optimized processing completed in {end_time - start_time:.4f} seconds")
    
    # Test pipeline
    pipeline = EpiphanyParallelPipeline(config)
    
    def stage1(data):
        x, y, t = data
        return (x * 2, y * 2, t)
    
    def stage2(data):
        x, y, t = data
        return (x + 1, y + 1, t + 0.1)
    
    pipeline_stages = [stage1, stage2, pixel_shader_func]
    
    start_time = __import__('time').time()
    pipeline_result = pipeline.execute_pipeline(test_data[:50], pipeline_stages)  # Smaller dataset for pipeline
    end_time = __import__('time').time()
    print(f"Pipeline completed in {end_time - start_time:.4f} seconds")
    print(f"Pipeline results length: {len(pipeline_result)}")
    
    print("Data parallel processing patterns for Epiphany created successfully!")


def example_usage():
    """Example of how to use the Epiphany data parallel patterns"""
    
    # Configuration for Epiphany system
    config = EpiphanyDataParallelConfig(
        num_cores=16,
        memory_per_core=32 * 1024,  # 32KB per core
        chunk_size=512
    )
    
    # Create the parallel processor
    processor = EpiphanyDataParallelProcessor(config)
    
    # Example: Processing vertex positions in parallel
    def transform_vertex(vertex_data):
        x, y, z, matrix = vertex_data
        # Apply transformation matrix (simplified)
        new_x = x * matrix[0] + y * matrix[1] + z * matrix[2]
        new_y = x * matrix[3] + y * matrix[4] + z * matrix[5] 
        new_z = x * matrix[6] + y * matrix[7] + z * matrix[8]
        return (new_x, new_y, new_z)
    
    # Create vertex data
    vertices = [(i, i*2, i*3, [1,0,0,0,1,0,0,0,1]) for i in range(1000)]
    
    # Process in parallel
    start_time = __import__('time').time()
    transformed_vertices = processor.parallel_map(transform_vertex, vertices)
    end_time = __import__('time').time()
    
    print(f"Processed {len(vertices)} vertices in {end_time - start_time:.4f} seconds")
    print(f"First vertex: {vertices[0]} -> {transformed_vertices[0]}")


if __name__ == "__main__":
    create_data_parallel_patterns()
    print("\n" + "="*50)
    example_usage()