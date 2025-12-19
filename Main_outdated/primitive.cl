// ============================================================================
// PE In-Cache Co-Processing - OpenCL Kernels
// Based on omnidb primitives, optimized for AMD APU
// ============================================================================

// Data types
typedef uint2 Record;

// Reduce operations
#define REDUCE_SUM 0
#define REDUCE_MAX 1
#define REDUCE_MIN 2
#define REDUCE_AVERAGE 3

// Constants
#define TEST_MAX (1 << 30)
#define TEST_MIN 0
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

// ============================================================================
// PE Prefetch Kernels - Run on dedicated CPU core
// These kernels touch data to bring it into shared LLC before GPU execution
// ============================================================================

// Prefetch kernel: reads data to load into LLC, signals completion via atomic
// Each work-item prefetches one chunk and signals when done
__kernel void prefetch_chunk_kernel(
    __global const char *data,         // Input data to prefetch
    __global volatile int *sync_flags, // Per-chunk completion flags
    int chunk_size,                    // Size of each chunk in bytes
    int total_size                     // Total data size
) {
  int chunk_id = get_global_id(0);
  int chunk_offset = chunk_id * chunk_size;

  // Bounds check
  if (chunk_offset >= total_size)
    return;

  int actual_chunk_size = min(chunk_size, total_size - chunk_offset);
  __global const char *chunk_ptr = data + chunk_offset;

  // Touch every cache line in this chunk to load into LLC
  // Read with stride of cache line size (64 bytes typical)
  volatile int sum = 0;
  for (int i = 0; i < actual_chunk_size; i += 64) {
    sum += chunk_ptr[i];
  }

  // Memory fence to ensure prefetch completes before signaling
  mem_fence(CLK_GLOBAL_MEM_FENCE);

  // Signal that this chunk is ready (atomic write)
  atomic_store_explicit((__global atomic_int *)&sync_flags[chunk_id], 1,
                        memory_order_release, memory_scope_device);
}

// Continuous prefetch kernel: prefetches chunks ahead of execution
// Runs on CPU, prefetches N chunks ahead
__kernel void prefetch_ahead_kernel(
    __global const char *data,         // Input data to prefetch
    __global volatile int *sync_flags, // Per-chunk completion flags
    __global volatile int
        *exec_progress,   // Current execution chunk (set by exec kernel)
    int chunk_size,       // Size of each chunk in bytes
    int total_chunks,     // Total number of chunks
    int prefetch_distance // How many chunks to prefetch ahead
) {
  // Single work-item continuously prefetches
  int prefetch_chunk = 0;

  while (prefetch_chunk < total_chunks) {
    // Check if we're too far ahead of execution
    int current_exec =
        atomic_load_explicit((__global atomic_int *)exec_progress,
                             memory_order_acquire, memory_scope_device);

    // Only prefetch if within distance of execution
    if (prefetch_chunk < current_exec + prefetch_distance) {
      int chunk_offset = prefetch_chunk * chunk_size;
      __global const char *chunk_ptr = data + chunk_offset;

      // Touch data to load into LLC
      volatile int sum = 0;
      int end = min(chunk_size, (total_chunks * chunk_size) - chunk_offset);
      for (int i = 0; i < end; i += 64) {
        sum += chunk_ptr[i];
      }

      mem_fence(CLK_GLOBAL_MEM_FENCE);

      // Signal chunk ready
      atomic_store_explicit((__global atomic_int *)&sync_flags[prefetch_chunk],
                            1, memory_order_release, memory_scope_device);
      prefetch_chunk++;
    }
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

inline int CONFLICT_FREE_OFFSET(int index) {
  return ((index) >> LOG_NUM_BANKS);
}

// ============================================================================
// Map Kernel
// Extracts x and y components from Record array into separate int arrays
// ============================================================================
__kernel void mapImpl_kernel(__global Record *d_R,    // Input records
                             int rLen,                // Number of records
                             __global int *d_output1, // Output: x components
                             __global int *d_output2  // Output: y components
) {
  int iGID = get_global_id(0);
  int delta = get_global_size(0);

  for (int pos = iGID; pos < rLen; pos += delta) {
    Record value = d_R[pos];
    d_output1[pos] = value.x;
    d_output2[pos] = value.y;
  }
}

// ============================================================================
// Filter Kernels
// Two-phase filter: mark matching records, then write them compactly
// ============================================================================

// Phase 1: Mark records that satisfy the filter condition
__kernel void filterImpl_map_kernel(
    __global Record *d_Rin, // Input records
    int beginPos,           // Start position
    int rLen,               // Number of records to process
    __global int *d_mark,   // Output: 1 if record matches, 0 otherwise
    int smallKey,           // Filter: key >= smallKey
    int largeKey,           // Filter: key <= largeKey
    __global int *d_temp    // Temporary buffer for x values
) {
  int iGID = get_global_id(0);
  int delta = get_global_size(0);

  for (int pos = iGID; pos < rLen; pos += delta) {
    int globalPos = beginPos + pos;
    Record value = d_Rin[globalPos];
    d_temp[globalPos] = value.x;

    int key = value.y;
    int flag = ((key >= smallKey) && (key <= largeKey)) ? 1 : 0;
    d_mark[globalPos] = flag;
  }
}

// ============================================================================
// PE-Synchronized Filter Kernel
// Waits for prefetch signal before processing each work-group's chunk
// ============================================================================
__kernel void filterImpl_map_PE_kernel(
    __global Record *d_Rin, // Input records
    int rLen,               // Total number of records
    __global int *d_mark,   // Output: 1 if record matches, 0 otherwise
    int smallKey,           // Filter: key >= smallKey
    int largeKey,           // Filter: key <= largeKey
    __global int *d_temp,   // Temporary buffer for x values
    __global volatile int *sync_flags, // Prefetch completion flags
    int records_per_chunk              // Records per prefetch chunk
) {
  int iGID = get_global_id(0);

  if (iGID >= rLen)
    return;

  // Determine which chunk this work-item belongs to
  int chunk_id = iGID / records_per_chunk;

  // Wait for prefetch of this chunk to complete
  // Spin until sync_flag is set
  while (atomic_load_explicit((__global atomic_int *)&sync_flags[chunk_id],
                              memory_order_acquire, memory_scope_device) == 0) {
    // Busy wait - prefetch kernel will set flag when chunk is in LLC
  }

  // Now process - data should be in LLC
  Record value = d_Rin[iGID];
  d_temp[iGID] = value.x;

  int key = value.y;
  int flag = ((key >= smallKey) && (key <= largeKey)) ? 1 : 0;
  d_mark[iGID] = flag;
}

// ============================================================================
// PE-Synchronized Map Kernel
// ============================================================================
__kernel void mapImpl_PE_kernel(
    __global Record *d_R,              // Input records
    int rLen,                          // Number of records
    __global int *d_output1,           // Output: x components
    __global int *d_output2,           // Output: y components
    __global volatile int *sync_flags, // Prefetch completion flags
    int records_per_chunk              // Records per prefetch chunk
) {
  int iGID = get_global_id(0);

  if (iGID >= rLen)
    return;

  // Determine which chunk this work-item belongs to
  int chunk_id = iGID / records_per_chunk;

  // Wait for prefetch of this chunk to complete
  while (atomic_load_explicit((__global atomic_int *)&sync_flags[chunk_id],
                              memory_order_acquire, memory_scope_device) == 0) {
    // Busy wait
  }

  // Process - data should be in LLC
  Record value = d_R[iGID];
  d_output1[iGID] = value.x;
  d_output2[iGID] = value.y;
}

// Phase 2: Write filtered records to output using scan results
__kernel void filterImpl_write_kernel(
    __global Record *d_Rout,    // Output records
    __global Record *d_Rin,     // Input records
    __global int *d_mark,       // Mark array (1 = keep, 0 = discard)
    __global int *d_markOutput, // Exclusive scan of marks (write positions)
    int beginPos,               // Start position
    int rLen                    // Number of records
) {
  int iGID = get_global_id(0);
  int delta = get_global_size(0);

  for (int pos = iGID; pos < rLen; pos += delta) {
    int flag = d_mark[pos];
    int writePos = d_markOutput[pos];

    if (flag) {
      d_Rout[writePos] = d_Rin[pos];
    }
  }
}

// ============================================================================
// Memset Kernel
// ============================================================================
__kernel void memset_int_kernel(__global int *d_R, int rLen, int value) {
  int iGID = get_global_id(0);
  int delta = get_global_size(0);

  for (int pos = iGID; pos < rLen; pos += delta) {
    d_R[pos] = value;
  }
}

// ============================================================================
// Prefix Sum (Scan) Kernels
// Based on Blelloch's efficient parallel scan algorithm
// ============================================================================

// Small array prefix sum (single work group)
__kernel void prefixSum_kernel(__global int *output, __global int *input,
                               __local int *block, const uint length) {
  int tid = get_local_id(0);
  int offset = 1;

  // Load data into local memory
  block[2 * tid] = input[2 * tid];
  block[2 * tid + 1] = input[2 * tid + 1];

  // Build sum in place up the tree
  for (int d = length >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      block[bi] += block[ai];
    }
    offset *= 2;
  }

  // Clear the last element
  if (tid == 0) {
    block[length - 1] = 0;
  }

  // Traverse down the tree building the scan in place
  for (int d = 1; d < length; d *= 2) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;

      int t = block[ai];
      block[ai] = block[bi];
      block[bi] += t;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Write results back to global memory
  output[2 * tid] = block[2 * tid];
  output[2 * tid + 1] = block[2 * tid + 1];
}

// Large array scan - processes blocks and stores block sums
__kernel void ScanLargeArrays_kernel(__global int *output, __global int *input,
                                     __local int *block, const uint block_size,
                                     const uint length,
                                     __global int *sumBuffer) {
  int tid = get_local_id(0);
  int gid = get_global_id(0);
  int bid = get_group_id(0);
  int offset = 1;

  // Load data into local memory
  block[2 * tid] = input[2 * gid];
  block[2 * tid + 1] = input[2 * gid + 1];

  // Build sum up the tree
  for (int d = block_size >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      block[bi] += block[ai];
    }
    offset *= 2;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Store block sum
  sumBuffer[bid] = block[block_size - 1];

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // Clear last element
  block[block_size - 1] = 0;

  // Traverse down tree
  for (int d = 1; d < block_size; d *= 2) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;

      int t = block[ai];
      block[ai] = block[bi];
      block[bi] += t;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Write results
  output[2 * gid] = block[2 * tid];
  output[2 * gid + 1] = block[2 * tid + 1];
}

// Add block sums back to scanned blocks
__kernel void blockAddition_kernel(__global int *input, // Block sums (scanned)
                                   __global int *output // Array to add to
) {
  int globalId = get_global_id(0);
  int groupId = get_group_id(0);
  int localId = get_local_id(0);

  __local int value[1];

  // One thread reads the block sum
  if (localId == 0) {
    value[0] = input[groupId];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  output[globalId] += value[0];
}

// ============================================================================
// Reduce Kernels
// Parallel reduction with different operations
// ============================================================================

__kernel void reduce_kernel(__global Record *d_input, __global int *d_output,
                            int rLen, int operation, __local int *s_data) {
  int tid = get_local_id(0);
  int gid = get_global_id(0);
  int blockDim = get_local_size(0);
  int gridDim = get_num_groups(0);
  int bid = get_group_id(0);

  // Initialize based on operation
  int initVal;
  switch (operation) {
  case REDUCE_SUM:
  case REDUCE_AVERAGE:
    initVal = 0;
    break;
  case REDUCE_MAX:
    initVal = TEST_MIN;
    break;
  case REDUCE_MIN:
    initVal = TEST_MAX;
    break;
  default:
    initVal = 0;
  }

  s_data[tid] = initVal;

  // Load and reduce within work group
  for (int i = gid; i < rLen; i += get_global_size(0)) {
    int val = d_input[i].y;

    switch (operation) {
    case REDUCE_SUM:
    case REDUCE_AVERAGE:
      s_data[tid] += val;
      break;
    case REDUCE_MAX:
      if (val > s_data[tid])
        s_data[tid] = val;
      break;
    case REDUCE_MIN:
      if (val < s_data[tid])
        s_data[tid] = val;
      break;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree reduction in shared memory
  for (int stride = blockDim / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      switch (operation) {
      case REDUCE_SUM:
      case REDUCE_AVERAGE:
        s_data[tid] += s_data[tid + stride];
        break;
      case REDUCE_MAX:
        if (s_data[tid + stride] > s_data[tid])
          s_data[tid] = s_data[tid + stride];
        break;
      case REDUCE_MIN:
        if (s_data[tid + stride] < s_data[tid])
          s_data[tid] = s_data[tid + stride];
        break;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write block result
  if (tid == 0) {
    d_output[bid] = s_data[0];
  }
}

// ============================================================================
// PE-Synchronized Reduce Kernel
// Waits for prefetch signal before processing each chunk
// ============================================================================
__kernel void
reduce_PE_kernel(__global Record *d_input, // Input records
                 __global int *d_output, // Output (partial sums per work-group)
                 int rLen,               // Total number of records
                 int operation,          // Reduce operation type
                 __global volatile int *sync_flags, // Prefetch completion flags
                 int records_per_chunk, // Records per prefetch chunk
                 __local int *s_data    // Local memory for reduction
) {
  int tid = get_local_id(0);
  int gid = get_global_id(0);
  int blockDim = get_local_size(0);
  int bid = get_group_id(0);

  // Initialize based on operation
  int initVal;
  switch (operation) {
  case REDUCE_SUM:
  case REDUCE_AVERAGE:
    initVal = 0;
    break;
  case REDUCE_MAX:
    initVal = TEST_MIN;
    break;
  case REDUCE_MIN:
    initVal = TEST_MAX;
    break;
  default:
    initVal = 0;
  }

  s_data[tid] = initVal;

  // Load and reduce - wait for each chunk's prefetch before accessing
  for (int i = gid; i < rLen; i += get_global_size(0)) {
    // Determine which chunk this element belongs to
    int chunk_id = i / records_per_chunk;

    // Wait for prefetch of this chunk
    while (atomic_load_explicit((__global atomic_int *)&sync_flags[chunk_id],
                                memory_order_acquire,
                                memory_scope_device) == 0) {
      // Busy wait
    }

    int val = d_input[i].y;

    switch (operation) {
    case REDUCE_SUM:
    case REDUCE_AVERAGE:
      s_data[tid] += val;
      break;
    case REDUCE_MAX:
      if (val > s_data[tid])
        s_data[tid] = val;
      break;
    case REDUCE_MIN:
      if (val < s_data[tid])
        s_data[tid] = val;
      break;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree reduction in shared memory
  for (int stride = blockDim / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      switch (operation) {
      case REDUCE_SUM:
      case REDUCE_AVERAGE:
        s_data[tid] += s_data[tid + stride];
        break;
      case REDUCE_MAX:
        if (s_data[tid + stride] > s_data[tid])
          s_data[tid] = s_data[tid + stride];
        break;
      case REDUCE_MIN:
        if (s_data[tid + stride] < s_data[tid])
          s_data[tid] = s_data[tid + stride];
        break;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write block result
  if (tid == 0) {
    d_output[bid] = s_data[0];
  }
}

// ============================================================================
// Scatter/Gather Kernels (for data movement)
// ============================================================================

__kernel void scatter_kernel(__global Record *d_input,
                             __global Record *d_output,
                             __global int *d_locations, int rLen) {
  int gid = get_global_id(0);
  int delta = get_global_size(0);

  for (int pos = gid; pos < rLen; pos += delta) {
    int targetLoc = d_locations[pos];
    d_output[targetLoc] = d_input[pos];
  }
}

__kernel void gather_kernel(__global Record *d_input, __global Record *d_output,
                            __global int *d_locations, int rLen) {
  int gid = get_global_id(0);
  int delta = get_global_size(0);

  for (int pos = gid; pos < rLen; pos += delta) {
    int sourceLoc = d_locations[pos];
    d_output[pos] = d_input[sourceLoc];
  }
}
