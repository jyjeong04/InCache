#include "primitives.h"
#include <algorithm>
#include <cstring>

// ============================================================================
// Kernel Objects
// ============================================================================
static cl::Kernel k_filterMap;
static cl::Kernel k_filterWrite;
static cl::Kernel k_map;
static cl::Kernel k_scan;
static cl::Kernel k_scanLargeArrays;
static cl::Kernel k_blockAddition;
static cl::Kernel k_prefixSum;
static cl::Kernel k_reduce;
static cl::Kernel k_memset;

// PE-specific kernels
static cl::Kernel k_prefetchChunk; // CPU prefetch kernel
static cl::Kernel k_filterMapPE;   // GPU filter with sync
static cl::Kernel k_mapPE;         // GPU map with sync
static cl::Kernel k_reducePE;      // GPU reduce with sync

static bool g_initialized = false;

// ============================================================================
// Work Group Configuration
// ============================================================================
static const size_t DEFAULT_LOCAL_SIZE = 256;
static const size_t SCAN_BLOCK_SIZE = 256;

// ============================================================================
// Initialization
// ============================================================================
void primitivesInit() {
  if (g_initialized) {
    return;
  }

  try {
    // Create kernels from compiled program
    k_filterMap = cl::Kernel(program, "filterImpl_map_kernel");
    k_filterWrite = cl::Kernel(program, "filterImpl_write_kernel");
    k_map = cl::Kernel(program, "mapImpl_kernel");
    k_scanLargeArrays = cl::Kernel(program, "ScanLargeArrays_kernel");
    k_blockAddition = cl::Kernel(program, "blockAddition_kernel");
    k_prefixSum = cl::Kernel(program, "prefixSum_kernel");
    k_memset = cl::Kernel(program, "memset_int_kernel");

    // PE-specific kernels (prefetch runs on CPU, execution on GPU)
    k_prefetchChunk = cl::Kernel(program, "prefetch_chunk_kernel");
    k_filterMapPE = cl::Kernel(program, "filterImpl_map_PE_kernel");
    k_mapPE = cl::Kernel(program, "mapImpl_PE_kernel");
    k_reducePE = cl::Kernel(program, "reduce_PE_kernel");

    g_initialized = true;
    std::cout << "[Primitives] Kernels initialized (including PE kernels)\n";

  } catch (cl::Error &err) {
    std::cerr << "[Primitives] Failed to create kernels: " << err.what() << " ("
              << err.err() << ")\n";
    throw;
  }
}

void primitivesCleanup() {
  if (!g_initialized) {
    return;
  }

  // Kernels are automatically released by destructor
  g_initialized = false;
}

// ============================================================================
// Helper Functions
// ============================================================================
size_t getOptimalWorkGroupSize(const cl::Device &device,
                               const cl::Kernel &kernel) {
  size_t maxWorkGroupSize =
      kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
  return std::min(maxWorkGroupSize, (size_t)DEFAULT_LOCAL_SIZE);
}

size_t roundUpToMultiple(size_t value, size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

// ============================================================================
// Prefix Sum Helper (used by filter)
// ============================================================================
static void exclusiveScan(cl::Buffer &d_input, cl::Buffer &d_output,
                          size_t numElements, cl::CommandQueue &queue) {

  // Simple implementation for now - can be optimized with multi-level scan
  size_t localSize = SCAN_BLOCK_SIZE;
  size_t globalSize = roundUpToMultiple(numElements / 2, localSize);

  // Temporary sum buffer for large arrays
  size_t numBlocks = (numElements + 2 * localSize - 1) / (2 * localSize);

  if (numBlocks <= 1) {

    // Small array - single pass scan
    k_prefixSum.setArg(0, d_output);
    k_prefixSum.setArg(1, d_input);
    k_prefixSum.setArg(2, cl::Local(2 * localSize * sizeof(int)));
    k_prefixSum.setArg(3, (cl_uint)numElements);

    queue.enqueueNDRangeKernel(k_prefixSum, cl::NullRange,
                               cl::NDRange(localSize), cl::NDRange(localSize));

  } else {
    cl::Buffer d_sumBuffer =
        cl_malloc(numBlocks * sizeof(int), CL_MEM_READ_WRITE);

    // First pass: scan each block

    k_scanLargeArrays.setArg(0, d_output);
    k_scanLargeArrays.setArg(1, d_input);
    k_scanLargeArrays.setArg(2, cl::Local(2 * localSize * sizeof(int)));
    k_scanLargeArrays.setArg(3, (cl_uint)(2 * localSize));
    k_scanLargeArrays.setArg(4, (cl_uint)numElements);
    k_scanLargeArrays.setArg(5, d_sumBuffer);

    queue.enqueueNDRangeKernel(k_scanLargeArrays, cl::NullRange,
                               cl::NDRange(globalSize), cl::NDRange(localSize));

    // Scan the sum buffer (recursive for very large arrays)
    if (numBlocks > 1) {
      cl::Buffer d_sumBufferScanned =
          cl_malloc(numBlocks * sizeof(int), CL_MEM_READ_WRITE);
      exclusiveScan(d_sumBuffer, d_sumBufferScanned, numBlocks, queue);

      // Add scanned sums back to each block

      k_blockAddition.setArg(0, d_sumBufferScanned);
      k_blockAddition.setArg(1, d_output);

      queue.enqueueNDRangeKernel(k_blockAddition, cl::NullRange,
                                 cl::NDRange(numElements),
                                 cl::NDRange(localSize));
    }
  }

  queue.finish();
}

// ============================================================================
// Filter Implementation
// ============================================================================
size_t filterBaseline(cl::Buffer &d_input, const void *h_input,
                      cl::Buffer &d_output, size_t numRecords, int smallKey,
                      int largeKey) {
  size_t dataSize = numRecords * sizeof(Record);

  // Copy input data to device (synchronous)
  execGpuQueue.enqueueWriteBuffer(d_input, CL_TRUE, 0, dataSize, h_input);

  // Allocate intermediate buffers
  cl::Buffer d_mark = cl_malloc(numRecords * sizeof(int), CL_MEM_READ_WRITE);
  cl::Buffer d_markScan =
      cl_malloc(numRecords * sizeof(int), CL_MEM_READ_WRITE);
  cl::Buffer d_temp = cl_malloc(numRecords * sizeof(int), CL_MEM_READ_WRITE);

  size_t localSize = getOptimalWorkGroupSize(GPU, k_filterMap);
  size_t globalSize = roundUpToMultiple(numRecords, localSize);

  // Step 1: Mark records that satisfy the filter condition
  k_filterMap.setArg(0, d_input);
  k_filterMap.setArg(1, 0); // beginPos
  k_filterMap.setArg(2, (int)numRecords);
  k_filterMap.setArg(3, d_mark);
  k_filterMap.setArg(4, smallKey);
  k_filterMap.setArg(5, largeKey);
  k_filterMap.setArg(6, d_temp);

  execGpuQueue.enqueueNDRangeKernel(k_filterMap, cl::NullRange,
                                    cl::NDRange(globalSize),
                                    cl::NDRange(localSize));

  // Step 2: Exclusive scan on marks to get write positions
  exclusiveScan(d_mark, d_markScan, numRecords, execGpuQueue);

  // Step 3: Write filtered records to output
  k_filterWrite.setArg(0, d_output);
  k_filterWrite.setArg(1, d_input);
  k_filterWrite.setArg(2, d_mark);
  k_filterWrite.setArg(3, d_markScan);
  k_filterWrite.setArg(4, 0); // beginPos
  k_filterWrite.setArg(5, (int)numRecords);

  execGpuQueue.enqueueNDRangeKernel(k_filterWrite, cl::NullRange,
                                    cl::NDRange(globalSize),
                                    cl::NDRange(localSize));

  execGpuQueue.finish();

  // Get output size
  int lastMark, lastScan;
  execGpuQueue.enqueueReadBuffer(
      d_mark, CL_TRUE, (numRecords - 1) * sizeof(int), sizeof(int), &lastMark);
  execGpuQueue.enqueueReadBuffer(d_markScan, CL_TRUE,
                                 (numRecords - 1) * sizeof(int), sizeof(int),
                                 &lastScan);

  return lastMark + lastScan;
}

size_t filterWithPE(cl::Buffer &d_input, const void *h_input,
                    cl::Buffer &d_output, size_t numRecords, int smallKey,
                    int largeKey) {
  size_t dataSize = numRecords * sizeof(Record);

  // Allocate intermediate buffers
  cl::Buffer d_mark = cl_malloc(numRecords * sizeof(int), CL_MEM_READ_WRITE);
  cl::Buffer d_markScan =
      cl_malloc(numRecords * sizeof(int), CL_MEM_READ_WRITE);
  cl::Buffer d_temp = cl_malloc(numRecords * sizeof(int), CL_MEM_READ_WRITE);

  // ========================================================================
  // PE In-Cache: CPU kernel touches data (loads to LLC), then GPU executes
  // ========================================================================
  // 1. Copy data to device memory
  // 2. CPU prefetch kernel reads data -> loads into shared LLC
  // 3. GPU execution kernel finds data in LLC -> reduced memory latency
  // ========================================================================

  // Copy data to device
  execGpuQueue.enqueueWriteBuffer(d_input, CL_TRUE, 0, dataSize, h_input);

  // CPU prefetch kernel: touch all data to load into LLC
  // Use multiple work-items to parallelize prefetching
  const size_t CHUNK_SIZE = peConfig.chunkSize;
  const size_t numChunks = (dataSize + CHUNK_SIZE - 1) / CHUNK_SIZE;

  k_prefetchChunk.setArg(0, d_input);
  k_prefetchChunk.setArg(1, d_input); // dummy sync_flags
  k_prefetchChunk.setArg(2, (int)CHUNK_SIZE);
  k_prefetchChunk.setArg(3, (int)dataSize);

  cl::Event prefetchEvent;
  prefetchQueue.enqueueNDRangeKernel(k_prefetchChunk, cl::NullRange,
                                     cl::NDRange(numChunks), cl::NullRange,
                                     nullptr, &prefetchEvent);
  prefetchEvent.wait(); // Ensure all data is in LLC

  // GPU execution - data should now be in LLC
  size_t localSize = getOptimalWorkGroupSize(GPU, k_filterMap);
  size_t globalSize = roundUpToMultiple(numRecords, localSize);

  k_filterMap.setArg(0, d_input);
  k_filterMap.setArg(1, 0); // beginPos
  k_filterMap.setArg(2, (int)numRecords);
  k_filterMap.setArg(3, d_mark);
  k_filterMap.setArg(4, smallKey);
  k_filterMap.setArg(5, largeKey);
  k_filterMap.setArg(6, d_temp);

  execGpuQueue.enqueueNDRangeKernel(k_filterMap, cl::NullRange,
                                    cl::NDRange(globalSize),
                                    cl::NDRange(localSize));
  execGpuQueue.finish();

  // Scan and write phases
  exclusiveScan(d_mark, d_markScan, numRecords, execGpuQueue);

  k_filterWrite.setArg(0, d_output);
  k_filterWrite.setArg(1, d_input);
  k_filterWrite.setArg(2, d_mark);
  k_filterWrite.setArg(3, d_markScan);
  k_filterWrite.setArg(4, 0);
  k_filterWrite.setArg(5, (int)numRecords);

  execGpuQueue.enqueueNDRangeKernel(k_filterWrite, cl::NullRange,
                                    cl::NDRange(globalSize),
                                    cl::NDRange(localSize));
  execGpuQueue.finish();

  // Get output size
  int lastMark, lastScan;
  execGpuQueue.enqueueReadBuffer(
      d_mark, CL_TRUE, (numRecords - 1) * sizeof(int), sizeof(int), &lastMark);
  execGpuQueue.enqueueReadBuffer(d_markScan, CL_TRUE,
                                 (numRecords - 1) * sizeof(int), sizeof(int),
                                 &lastScan);

  return lastMark + lastScan;
}

// ============================================================================
// Map Implementation
// ============================================================================
void mapBaseline(cl::Buffer &d_input, const void *h_input,
                 cl::Buffer &d_output1, cl::Buffer &d_output2,
                 size_t numRecords) {
  size_t dataSize = numRecords * sizeof(Record);

  // Copy input to device
  execGpuQueue.enqueueWriteBuffer(d_input, CL_TRUE, 0, dataSize, h_input);

  size_t localSize = getOptimalWorkGroupSize(GPU, k_map);
  size_t globalSize = roundUpToMultiple(numRecords, localSize);

  k_map.setArg(0, d_input);
  k_map.setArg(1, (int)numRecords);
  k_map.setArg(2, d_output1);
  k_map.setArg(3, d_output2);

  execGpuQueue.enqueueNDRangeKernel(
      k_map, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize));
  execGpuQueue.finish();
}

void mapWithPE(cl::Buffer &d_input, const void *h_input, cl::Buffer &d_output1,
               cl::Buffer &d_output2, size_t numRecords) {
  size_t dataSize = numRecords * sizeof(Record);

  // ========================================================================
  // PE: CPU kernel prefetches chunks, GPU kernel waits for sync_flags
  // ========================================================================
  const size_t CHUNK_SIZE = peConfig.chunkSize;
  const size_t recordsPerChunk = CHUNK_SIZE / sizeof(Record);
  const size_t numChunks = (numRecords + recordsPerChunk - 1) / recordsPerChunk;

  // Create and initialize sync_flags
  cl::Buffer d_syncFlags =
      cl_malloc(numChunks * sizeof(int), CL_MEM_READ_WRITE);
  int zero = 0;
  k_memset.setArg(0, d_syncFlags);
  k_memset.setArg(1, zero);
  k_memset.setArg(2, (int)numChunks);
  execGpuQueue.enqueueNDRangeKernel(k_memset, cl::NullRange,
                                    cl::NDRange(numChunks), cl::NullRange);
  execGpuQueue.finish();

  // Copy input data to device
  execGpuQueue.enqueueWriteBuffer(d_input, CL_TRUE, 0, dataSize, h_input);

  size_t localSize = getOptimalWorkGroupSize(GPU, k_mapPE);
  size_t globalSize = roundUpToMultiple(numRecords, localSize);

  // Launch Prefetch kernel on CPU
  k_prefetchChunk.setArg(0, d_input);
  k_prefetchChunk.setArg(1, d_syncFlags);
  k_prefetchChunk.setArg(2, (int)CHUNK_SIZE);
  k_prefetchChunk.setArg(3, (int)dataSize);

  cl::Event prefetchEvent;
  prefetchQueue.enqueueNDRangeKernel(k_prefetchChunk, cl::NullRange,
                                     cl::NDRange(numChunks), cl::NullRange,
                                     nullptr, &prefetchEvent);

  // Launch Map kernel on GPU
  k_mapPE.setArg(0, d_input);
  k_mapPE.setArg(1, (int)numRecords);
  k_mapPE.setArg(2, d_output1);
  k_mapPE.setArg(3, d_output2);
  k_mapPE.setArg(4, d_syncFlags);
  k_mapPE.setArg(5, (int)recordsPerChunk);

  cl::Event mapEvent;
  execGpuQueue.enqueueNDRangeKernel(k_mapPE, cl::NullRange,
                                    cl::NDRange(globalSize),
                                    cl::NDRange(localSize), nullptr, &mapEvent);

  // Wait for both kernels
  prefetchEvent.wait();
  mapEvent.wait();
}

// ============================================================================
// Scan Implementation
// ============================================================================
void scanBaseline(cl::Buffer &d_input, cl::Buffer &d_output, size_t numRecords,
                  int operation) {
  exclusiveScan(d_input, d_output, numRecords, execGpuQueue);
}

void scanWithPE(cl::Buffer &d_input, const void *h_input, cl::Buffer &d_output,
                size_t numRecords, int operation) {
  // For scan, prefetch the entire input first since scan has dependencies
  prefetchToDevice(d_input, h_input, numRecords * sizeof(Record));
  exclusiveScan(d_input, d_output, numRecords, execGpuQueue);
}

// ============================================================================
// Reduce Implementation
// ============================================================================
int reduceBaseline(cl::Buffer &d_input, const void *h_input, size_t numRecords,
                   int operation) {
  size_t dataSize = numRecords * sizeof(Record);

  // Copy input to device
  execGpuQueue.enqueueWriteBuffer(d_input, CL_TRUE, 0, dataSize, h_input);

  // Simple CPU-side reduction for now
  // In production, use GPU reduction kernel
  Record *h_data = (Record *)host_malloc(dataSize);
  execGpuQueue.enqueueReadBuffer(d_input, CL_TRUE, 0, dataSize, h_data);

  long long result = 0;
  switch (operation) {
  case REDUCE_SUM: {
    long long sum = 0;
    for (size_t i = 0; i < numRecords; i++) {
      sum += h_data[i].s[1];
    }
    result = sum /
             (long long)
                 numRecords; // Return average to prevent overflow (same as PE)
    break;
  }
  case REDUCE_MAX: {
    result = h_data[0].s[1];
    for (size_t i = 1; i < numRecords; i++) {
      if ((int)h_data[i].s[1] > result)
        result = h_data[i].s[1];
    }
    break;
  }
  case REDUCE_MIN: {
    result = h_data[0].s[1];
    for (size_t i = 1; i < numRecords; i++) {
      if ((int)h_data[i].s[1] < result)
        result = h_data[i].s[1];
    }
    break;
  }
  case REDUCE_AVERAGE: {
    long long sum = 0;
    for (size_t i = 0; i < numRecords; i++) {
      sum += h_data[i].s[1];
    }
    result = (int)(sum / numRecords);
    break;
  }
  }

  host_free(h_data);
  return (int)result;
}

int reduceWithPE(cl::Buffer &d_input, const void *h_input, size_t numRecords,
                 int operation) {
  size_t dataSize = numRecords * sizeof(Record);

  // ========================================================================
  // PE: CPU kernel prefetches chunks, GPU kernel waits for sync_flags
  // ========================================================================
  const size_t CHUNK_SIZE = peConfig.chunkSize;
  const size_t recordsPerChunk = CHUNK_SIZE / sizeof(Record);
  const size_t numChunks = (numRecords + recordsPerChunk - 1) / recordsPerChunk;

  // Create and initialize sync_flags
  cl::Buffer d_syncFlags =
      cl_malloc(numChunks * sizeof(int), CL_MEM_READ_WRITE);
  int zero = 0;
  k_memset.setArg(0, d_syncFlags);
  k_memset.setArg(1, zero);
  k_memset.setArg(2, (int)numChunks);
  execGpuQueue.enqueueNDRangeKernel(k_memset, cl::NullRange,
                                    cl::NDRange(numChunks), cl::NullRange);
  execGpuQueue.finish();

  // Copy input data to device
  execGpuQueue.enqueueWriteBuffer(d_input, CL_TRUE, 0, dataSize, h_input);

  // Launch Prefetch kernel on CPU
  k_prefetchChunk.setArg(0, d_input);
  k_prefetchChunk.setArg(1, d_syncFlags);
  k_prefetchChunk.setArg(2, (int)CHUNK_SIZE);
  k_prefetchChunk.setArg(3, (int)dataSize);

  cl::Event prefetchEvent;
  prefetchQueue.enqueueNDRangeKernel(k_prefetchChunk, cl::NullRange,
                                     cl::NDRange(numChunks), cl::NullRange,
                                     nullptr, &prefetchEvent);

  // Setup and launch reduce kernel on GPU
  size_t localSize = SCAN_BLOCK_SIZE;
  size_t numGroups = (numRecords + localSize - 1) / localSize;
  numGroups = std::min(numGroups, (size_t)256);
  size_t globalSize = numGroups * localSize;

  // Allocate output buffer for partial results
  cl::Buffer d_partial = cl_malloc(numGroups * sizeof(int), CL_MEM_READ_WRITE);

  k_reducePE.setArg(0, d_input);
  k_reducePE.setArg(1, d_partial);
  k_reducePE.setArg(2, (int)numRecords);
  k_reducePE.setArg(3, operation);
  k_reducePE.setArg(4, d_syncFlags);
  k_reducePE.setArg(5, (int)recordsPerChunk);
  k_reducePE.setArg(6, cl::Local(localSize * sizeof(int)));

  cl::Event reduceEvent;
  execGpuQueue.enqueueNDRangeKernel(
      k_reducePE, cl::NullRange, cl::NDRange(globalSize),
      cl::NDRange(localSize), nullptr, &reduceEvent);

  // Wait for both kernels
  prefetchEvent.wait();
  reduceEvent.wait();

  // Read partial results and combine on CPU
  std::vector<int> partialResults(numGroups);
  execGpuQueue.enqueueReadBuffer(d_partial, CL_TRUE, 0, numGroups * sizeof(int),
                                 partialResults.data());

  long long result = 0;
  switch (operation) {
  case REDUCE_SUM:
  case REDUCE_AVERAGE: {
    long long sum = 0;
    for (size_t i = 0; i < numGroups; i++) {
      sum += partialResults[i];
    }
    result = sum / (long long)numRecords;
    break;
  }
  case REDUCE_MAX: {
    result = partialResults[0];
    for (size_t i = 1; i < numGroups; i++) {
      if (partialResults[i] > result)
        result = partialResults[i];
    }
    break;
  }
  case REDUCE_MIN: {
    result = partialResults[0];
    for (size_t i = 1; i < numGroups; i++) {
      if (partialResults[i] < result)
        result = partialResults[i];
    }
    break;
  }
  }

  return (int)result;
}
