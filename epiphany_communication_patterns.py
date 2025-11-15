#!/usr/bin/env python3
"""
Communication Patterns Between Epiphany Cores
Implements efficient communication patterns for Epiphany's distributed architecture
"""

import numpy as np
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import threading
import time
import struct


@dataclass
class CommunicationConfig:
    """Configuration for core communication"""
    num_cores: int = 16
    rows: int = 4
    cols: int = 4
    use_mailbox: bool = True
    use_shared_memory: bool = True
    max_message_size: int = 1024  # bytes
    enable_dma: bool = True
    sync_method: str = "barrier"  # "barrier", "flag", "semaphore"


class EpiphanyCoreCommunicator:
    """Manages communication between Epiphany cores"""
    
    def __init__(self, config: CommunicationConfig = None):
        self.config = config or CommunicationConfig()
        self.num_cores = self.config.num_cores
        self.rows = self.config.rows
        self.cols = self.config.cols
        self.use_mailbox = self.config.use_mailbox
        self.use_shared_memory = self.config.use_shared_memory
        self.max_message_size = self.config.max_message_size
        self.enable_dma = self.config.enable_dma
        self.sync_method = self.config.sync_method
        
        # Simulated communication infrastructure
        self.mailboxes = [{} for _ in range(self.num_cores)]
        self.shared_memory = bytearray(64 * 1024)  # 64KB shared memory
        self.core_flags = [False] * self.num_cores
        self.barrier_count = 0
        self.barrier_lock = threading.Lock()
        self.barrier_event = threading.Event()
    
    def send_message(self, from_core: int, to_core: int, message: Any):
        """Send a message from one core to another"""
        if self.use_mailbox:
            # Serialize message
            serialized_msg = self._serialize_message(message)
            
            # Put message in recipient's mailbox
            with threading.Lock():
                if to_core not in self.mailboxes[from_core]:
                    self.mailboxes[from_core][to_core] = []
                self.mailboxes[from_core][to_core].append(serialized_msg)
        else:
            # Use shared memory for communication
            self._write_to_shared_memory(from_core, to_core, message)
    
    def receive_message(self, core_id: int, from_core: Optional[int] = None) -> Optional[Any]:
        """Receive a message from another core"""
        if self.use_mailbox:
            with threading.Lock():
                if from_core is not None:
                    # Receive from specific core
                    if from_core in self.mailboxes[core_id] and self.mailboxes[core_id][from_core]:
                        msg = self.mailboxes[core_id][from_core].pop(0)
                        return self._deserialize_message(msg)
                else:
                    # Receive from any core
                    for sender, messages in self.mailboxes[core_id].items():
                        if messages:
                            msg = messages.pop(0)
                            return self._deserialize_message(msg)
        else:
            # Read from shared memory
            return self._read_from_shared_memory(core_id, from_core)
        
        return None
    
    def _serialize_message(self, message: Any) -> bytes:
        """Serialize a message for transmission"""
        try:
            # Try to handle different types of messages
            if isinstance(message, (int, float)):
                return struct.pack('f', float(message))
            elif isinstance(message, (list, tuple)):
                # Handle arrays of numbers
                fmt = 'f' * len(message)
                return struct.pack(fmt, *message)
            elif isinstance(message, str):
                return message.encode('utf-8')
            else:
                # For other types, convert to string representation
                return str(message).encode('utf-8')
        except:
            # Fallback serialization
            return str(message).encode('utf-8')
    
    def _deserialize_message(self, message_bytes: bytes) -> Any:
        """Deserialize a received message"""
        try:
            # Try to determine message type and deserialize appropriately
            if len(message_bytes) == 4:  # Might be a single float
                return struct.unpack('f', message_bytes)[0]
            elif len(message_bytes) % 4 == 0:  # Might be array of floats
                fmt = 'f' * (len(message_bytes) // 4)
                return list(struct.unpack(fmt, message_bytes))
            else:
                # Try to decode as string
                return message_bytes.decode('utf-8')
        except:
            # Fallback deserialization
            try:
                return message_bytes.decode('utf-8')
            except:
                return str(message_bytes)
    
    def _write_to_shared_memory(self, from_core: int, to_core: int, message: Any):
        """Write message to shared memory"""
        if not self.use_shared_memory:
            return
        
        # Calculate position in shared memory based on cores
        # This is a simplified approach - real implementation would use proper addressing
        offset = ((from_core * self.rows + to_core) * 256) % len(self.shared_memory)
        
        serialized = self._serialize_message(message)
        if len(serialized) <= len(self.shared_memory) - offset:
            self.shared_memory[offset:offset+len(serialized)] = serialized
    
    def _read_from_shared_memory(self, core_id: int, from_core: Optional[int]) -> Optional[Any]:
        """Read message from shared memory"""
        if not self.use_shared_memory:
            return None
        
        # Calculate position to read from
        if from_core is not None:
            offset = ((from_core * self.rows + core_id) * 256) % len(self.shared_memory)
            # Read from calculated offset
            data = self.shared_memory[offset:offset+256]  # Read 256 bytes
            # Find end of message (simplified)
            end_idx = data.find(b'\x00')  # Null terminator
            if end_idx != -1:
                data = data[:end_idx]
            return self._deserialize_message(data)
        
        return None
    
    def broadcast_message(self, from_core: int, message: Any):
        """Broadcast a message to all other cores"""
        for to_core in range(self.num_cores):
            if to_core != from_core:
                self.send_message(from_core, to_core, message)
    
    def gather_from_cores(self, receiving_core: int, num_values: int) -> List[Any]:
        """Gather values from all cores"""
        gathered_values = []
        for core_id in range(self.num_cores):
            if core_id != receiving_core:
                # Receive value from this core
                value = self.receive_message(receiving_core, core_id)
                if value is not None:
                    gathered_values.append(value)
        
        return gathered_values
    
    def scatter_to_cores(self, from_core: int, values: List[Any]):
        """Scatter values to all other cores"""
        if len(values) != self.num_cores - 1:
            # Pad or truncate values if necessary
            while len(values) < self.num_cores - 1:
                values.append(values[-1] if values else 0)
            values = values[:self.num_cores - 1]
        
        for i, to_core in enumerate(range(self.num_cores)):
            if to_core != from_core and i < len(values):
                self.send_message(from_core, to_core, values[i])
    
    def synchronize_cores(self):
        """Synchronize all cores using the configured method"""
        if self.sync_method == "barrier":
            self._barrier_synchronization()
        elif self.sync_method == "flag":
            self._flag_synchronization()
        elif self.sync_method == "semaphore":
            # Simplified semaphore implementation
            self.core_flags = [False] * self.num_cores
        # For "none", no synchronization is performed
    
    def _barrier_synchronization(self):
        """Barrier synchronization - all cores wait until all reach the barrier"""
        with self.barrier_lock:
            self.barrier_count += 1
            if self.barrier_count == self.num_cores:
                self.barrier_count = 0
                self.barrier_event.set()
            else:
                event = self.barrier_event
                self.barrier_lock.release()
                event.wait()
                self.barrier_lock.acquire()
                if self.barrier_count == 0:
                    self.barrier_event.clear()
    
    def _flag_synchronization(self):
        """Flag-based synchronization"""
        # Set flag for this core
        core_id = threading.current_thread().ident % self.num_cores
        self.core_flags[core_id] = True
        
        # Wait until all cores have set their flags
        time.sleep(0.001)  # Brief sleep to allow other cores to catch up
        while not all(self.core_flags):
            time.sleep(0.001)


class CommunicationPatternLibrary:
    """Library of common communication patterns for Epiphany"""
    
    def __init__(self, communicator: EpiphanyCoreCommunicator):
        self.comm = communicator
    
    def ring_communication(self, core_id: int, data: Any, operation: str = "forward") -> Any:
        """Implement ring communication pattern"""
        next_core = (core_id + 1) % self.comm.num_cores
        prev_core = (core_id - 1) % self.comm.num_cores
        
        # Send data to next core
        if operation == "forward":
            self.comm.send_message(core_id, next_core, data)
            # Optionally receive data from previous core
            received_data = self.comm.receive_message(core_id, prev_core)
            return received_data
        elif operation == "collect":
            # Collect data from previous core, add own data, send to next
            prev_data = self.comm.receive_message(core_id, prev_core)
            if prev_data is not None:
                if isinstance(prev_data, list):
                    prev_data.append(data)
                else:
                    prev_data = [prev_data, data]
            else:
                prev_data = [data]
            
            self.comm.send_message(core_id, next_core, prev_data)
            return prev_data
    
    def tree_communication(self, core_id: int, data: Any, operation: str = "sum") -> Any:
        """Implement tree-based communication pattern"""
        parent_core = (core_id - 1) // 2 if core_id > 0 else -1
        left_child = core_id * 2 + 1 if core_id * 2 + 1 < self.comm.num_cores else -1
        right_child = core_id * 2 + 2 if core_id * 2 + 2 < self.comm.num_cores else -1
        
        # Handle children first
        child_results = []
        if left_child != -1:
            child_data = self.comm.receive_message(core_id, left_child)
            if child_data is not None:
                child_results.append(child_data)
        if right_child != -1:
            child_data = self.comm.receive_message(core_id, right_child)
            if child_data is not None:
                child_results.append(child_data)
        
        # Combine results based on operation
        result = data
        for child_result in child_results:
            if operation == "sum":
                result += child_result
            elif operation == "max":
                result = max(result, child_result)
            elif operation == "min":
                result = min(result, child_result)
            elif operation == "avg":
                result = (result + child_result) / 2
        
        # Send result to parent
        if parent_core != -1 and result is not None:
            self.comm.send_message(core_id, parent_core, result)
        
        return result
    
    def butterfly_communication(self, core_id: int, data: Any) -> Any:
        """Implement butterfly communication pattern"""
        # Butterfly pattern: cores communicate with partners that change each round
        stages = int(np.log2(self.comm.num_cores))
        partner = core_id
        
        for stage in range(stages):
            # Calculate partner for this stage
            mask = 1 << stage
            partner = core_id ^ mask
            
            if partner < self.comm.num_cores:
                # Send data to partner
                self.comm.send_message(core_id, partner, data)
                
                # Receive data from partner
                received = self.comm.receive_message(core_id, partner)
                if received is not None:
                    # Combine data (example: averaging)
                    if isinstance(data, (int, float)) and isinstance(received, (int, float)):
                        data = (data + received) / 2
                    elif isinstance(data, list) and isinstance(received, list):
                        # Combine lists
                        data = data + received
                    else:
                        # Just keep original data
                        pass
        
        return data


class EpiphanyCommunicationManager:
    """Complete communication management system for Epiphany"""
    
    def __init__(self, config: CommunicationConfig = None):
        self.config = config or CommunicationConfig()
        self.communicator = EpiphanyCoreCommunicator(self.config)
        self.pattern_lib = CommunicationPatternLibrary(self.communicator)
        self.core_threads = []
    
    def start_core_simulation(self, core_id: int, task_func: Callable[[int], Any]):
        """Simulate a core running with communication capabilities"""
        def core_task():
            # Set up core-specific communication context
            result = task_func(core_id)
            # Perform any necessary post-task communication
            return result
        
        thread = threading.Thread(target=core_task)
        self.core_threads.append(thread)
        thread.start()
        return thread
    
    def execute_collective_communication(self, operation: str, data: List[Any]) -> List[Any]:
        """Execute collective communication operations"""
        if operation == "broadcast":
            # Use the first item in data as the broadcast value
            if data:
                self.communicator.broadcast_message(0, data[0])
            return [data[0]] * self.config.num_cores if data else []
        elif operation == "scatter":
            # Distribute data items to different cores
            self.communicator.scatter_to_cores(0, data)
            return data
        elif operation == "gather":
            # Collect results from all cores
            return self.communicator.gather_from_cores(0, len(data))
        else:
            return data


def create_communication_patterns():
    """Create communication patterns between Epiphany cores"""
    print("Creating communication patterns between Epiphany cores...")
    
    # Create communication configuration
    config = CommunicationConfig(
        num_cores=16,
        use_mailbox=True,
        use_shared_memory=True,
        enable_dma=True
    )
    
    # Create communicator
    communicator = EpiphanyCoreCommunicator(config)
    
    # Test basic point-to-point communication
    print("Testing point-to-point communication...")
    communicator.send_message(0, 1, "Hello from core 0")
    received = communicator.receive_message(1, 0)
    print(f"Core 1 received: {received}")
    
    # Test broadcast
    print("Testing broadcast communication...")
    communicator.broadcast_message(0, "Broadcast message")
    # Simulate receiving on a few cores
    for i in range(1, 5):
        received = communicator.receive_message(i, 0)
        if received:
            print(f"Core {i} received broadcast: {received}")
    
    # Test pattern library
    print("Testing communication patterns...")
    pattern_lib = CommunicationPatternLibrary(communicator)
    
    # Ring communication test
    for i in range(5):
        result = pattern_lib.ring_communication(i, f"data_from_core_{i}")
        print(f"Ring comm result for core {i}: {result}")
    
    # Tree communication test
    tree_result = pattern_lib.tree_communication(0, 10, "sum")
    print(f"Tree communication result: {tree_result}")
    
    # Butterfly communication test
    butterfly_result = pattern_lib.butterfly_communication(0, [1, 2, 3])
    print(f"Butterfly communication result: {butterfly_result}")
    
    # Create communication manager
    comm_manager = EpiphanyCommunicationManager(config)
    
    # Test collective operations
    print("Testing collective communication operations...")
    broadcast_result = comm_manager.execute_collective_communication("broadcast", ["shared_data"])
    print(f"Broadcast result: {broadcast_result[:3]}...")  # Show first 3
    
    # Example: Simulate cores working together on a task
    def core_task(core_id):
        """Example task that a core might perform with communication"""
        local_data = core_id * 10  # Local computation
        
        # Synchronize with other cores
        communicator.synchronize_cores()
        
        # Send own data to next core in ring
        next_core = (core_id + 1) % config.num_cores
        communicator.send_message(core_id, next_core, local_data)
        
        # Receive data from previous core
        prev_core = (core_id - 1) % config.num_cores
        received_data = communicator.receive_message(core_id, prev_core)
        
        # Return combined result
        return {
            "core_id": core_id,
            "local_data": local_data,
            "received_data": received_data,
            "timestamp": time.time()
        }
    
    # Simulate core tasks
    print("Simulating cores with communication...")
    results = []
    for i in range(min(4, config.num_cores)):  # Only simulate a few cores for demo
        result = core_task(i)
        results.append(result)
        print(f"Core {i} result: {result}")
    
    # Create a C header file with communication patterns
    c_header_code = '''/* Communication Patterns for Epiphany Architecture */
#ifndef EPIPHANY_COMMUNICATION_H
#define EPIPHANY_COMMUNICATION_H

#include <e_lib.h>

/* Core identification */
#define CORE_ROWS 4
#define CORE_COLS 4
#define NUM_CORES (CORE_ROWS * CORE_COLS)

/* Communication modes */
#define COMM_MAILBOX 1
#define COMM_SHARED_MEM 2
#define COMM_DMA 3

/* Message structure */
typedef struct {
    unsigned char type;
    unsigned char src_core;
    unsigned char dst_core;
    unsigned short length;
    volatile char data[252];  // Max 256 bytes including header
} e_message_t;

/* Communication flags */
extern volatile unsigned int e_comm_flags[NUM_CORES];
extern volatile e_message_t e_shared_msg_buffer[256];

/* Function prototypes */
int e_send_message(unsigned char dst_core, void* data, unsigned short length);
int e_receive_message(unsigned char src_core, void* buffer, unsigned short max_length);
int e_broadcast_message(void* data, unsigned short length);
void e_synchronize_cores();
void e_barrier_sync();
int e_ring_send(unsigned char next_core, void* data, unsigned short length);
int e_ring_receive(unsigned char prev_core, void* buffer, unsigned short max_length);

/* Mailbox communication functions */
static inline int e_mbox_put(unsigned char core, void* data, unsigned short length) {
    // Implementation would use Epiphany mailboxes
    return 0;
}

static inline int e_mbox_get(unsigned char core, void* buffer, unsigned short max_length) {
    // Implementation would use Epiphany mailboxes
    return 0;
}

/* Shared memory communication functions */
static inline volatile void* e_get_shared_mem_ptr(unsigned int offset) {
    // Epiphany shared memory starts at 0x8f000000
    return (volatile void*)(0x8f000000 + offset);
}

/* DMA communication functions (if available) */
#if defined(E_ENABLE_DMA)
static inline int e_dma_copy(volatile void* dst, volatile void* src, unsigned int length) {
    // Implementation would use Epiphany DMA engine
    return 0;
}
#endif

#endif /* EPIPHANY_COMMUNICATION_H */
'''
    
    with open("epiphany_communication.h", "w") as f:
        f.write(c_header_code)
    print("✓ Generated epiphany_communication.h")
    
    # Create an example implementation file
    c_impl_code = '''/* Implementation of Epiphany Communication Patterns */
#include "epiphany_communication.h"
#include <string.h>

/* Global variables for communication */
volatile unsigned int e_comm_flags[NUM_CORES] = {0};
volatile e_message_t e_shared_msg_buffer[256] = {{0}};

int e_send_message(unsigned char dst_core, void* data, unsigned short length) {
    if (length > 252) return -1; // Message too large
    
    unsigned row, col;
    e_get_coords(&row, &col);
    unsigned char src_core = row * CORE_COLS + col;
    
    // Find an available slot in the shared message buffer
    for (int i = 0; i < 256; i++) {
        if (e_shared_msg_buffer[i].type == 0) { // Empty slot
            e_shared_msg_buffer[i].type = 1; // Mark as used
            e_shared_msg_buffer[i].src_core = src_core;
            e_shared_msg_buffer[i].dst_core = dst_core;
            e_shared_msg_buffer[i].length = length;
            
            if (data != NULL && length > 0) {
                for (int j = 0; j < length; j++) {
                    e_shared_msg_buffer[i].data[j] = ((char*)data)[j];
                }
            }
            
            // Set flag indicating message is ready
            e_comm_flags[dst_core] = 1;
            return 0;
        }
    }
    
    return -1; // No available slots
}

int e_receive_message(unsigned char src_core, void* buffer, unsigned short max_length) {
    if (buffer == NULL) return -1;
    
    // Look for messages from src_core
    for (int i = 0; i < 256; i++) {
        if (e_shared_msg_buffer[i].type == 1 && 
            e_shared_msg_buffer[i].src_core == src_core) {
            
            unsigned short msg_len = e_shared_msg_buffer[i].length;
            if (msg_len > max_length) return -1; // Buffer too small
            
            for (int j = 0; j < msg_len; j++) {
                ((char*)buffer)[j] = e_shared_msg_buffer[i].data[j];
            }
            
            // Mark message as processed
            e_shared_msg_buffer[i].type = 0;
            return msg_len;
        }
    }
    
    return 0; // No message found
}

void e_synchronize_cores() {
    // Simple barrier implementation using core flags
    unsigned row, col;
    e_get_coords(&row, &col);
    unsigned char core_id = row * CORE_COLS + col;
    
    // Set flag for this core
    e_comm_flags[core_id] = 1;
    
    // Wait for all cores to set their flags
    unsigned all_ready = 0;
    do {
        all_ready = 1;
        for (int i = 0; i < NUM_CORES; i++) {
            if (e_comm_flags[i] == 0) {
                all_ready = 0;
                break;
            }
        }
    } while (!all_ready);
}

int e_broadcast_message(void* data, unsigned short length) {
    unsigned row, col;
    e_get_coords(&row, &col);
    unsigned char src_core = row * CORE_COLS + col;
    
    // Send to all other cores
    for (int i = 0; i < NUM_CORES; i++) {
        if (i != src_core) {
            if (e_send_message(i, data, length) != 0) {
                return -1;
            }
        }
    }
    
    return 0;
}

/* Ring communication functions */
int e_ring_send(unsigned char next_core, void* data, unsigned short length) {
    return e_send_message(next_core, data, length);
}

int e_ring_receive(unsigned char prev_core, void* buffer, unsigned short max_length) {
    return e_receive_message(prev_core, buffer, max_length);
}
'''
    
    with open("epiphany_communication.c", "w") as f:
        f.write(c_impl_code)
    print("✓ Generated epiphany_communication.c")
    
    print("Communication patterns between Epiphany cores created successfully!")


def example_usage():
    """Example of how to use the communication patterns"""
    
    # Configuration for communication system
    config = CommunicationConfig(
        num_cores=4,  # Smaller number for example
        use_mailbox=True,
        sync_method="barrier"
    )
    
    # Create the communication manager
    comm_manager = EpiphanyCommunicationManager(config)
    
    # Example: Parallel reduction using communication
    def parallel_sum_core(core_id):
        """Example core function that performs part of a parallel sum"""
        # Each core has its own value to contribute
        local_value = core_id * 10 + 5
        
        print(f"Core {core_id}: starting with value {local_value}")
        
        # In a real implementation, cores would communicate
        # to perform the reduction, but here we just return the value
        return local_value
    
    # Simulate the parallel sum
    results = []
    for i in range(config.num_cores):
        result = parallel_sum_core(i)
        results.append(result)
        print(f"Core {i} result: {result}")
    
    total = sum(results)
    print(f"Sum of all core values: {total}")


if __name__ == "__main__":
    create_communication_patterns()
    print("\n" + "="*50)
    example_usage()