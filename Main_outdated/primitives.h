#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "common.h"
#include "prefetch.h"

// ============================================================================
// Primitive Operations
// ============================================================================

// Initialize primitives (creates kernels)
void primitivesInit();

// Cleanup primitives
void primitivesCleanup();

// ============================================================================
// Filter Operation
// ============================================================================
// Filter records where smallKey <= record.y <= largeKey
// Returns the number of output records

// Baseline: GPU-only execution without prefetching
size_t filterBaseline(cl::Buffer &d_input, const void *h_input,
                      cl::Buffer &d_output, size_t numRecords,
                      int smallKey, int largeKey);

// PE optimized: CPU prefetches while GPU executes
size_t filterWithPE(cl::Buffer &d_input, const void *h_input,
                    cl::Buffer &d_output, size_t numRecords,
                    int smallKey, int largeKey);

// ============================================================================
// Map Operation  
// ============================================================================
// Extract x and y components from Record into separate arrays

// Baseline: GPU-only execution
void mapBaseline(cl::Buffer &d_input, const void *h_input,
                 cl::Buffer &d_output1, cl::Buffer &d_output2,
                 size_t numRecords);

// PE optimized
void mapWithPE(cl::Buffer &d_input, const void *h_input,
               cl::Buffer &d_output1, cl::Buffer &d_output2,
               size_t numRecords);

// ============================================================================
// Scan (Prefix Sum) Operation
// ============================================================================
// Exclusive prefix sum on the y component of records

// Baseline
void scanBaseline(cl::Buffer &d_input, cl::Buffer &d_output,
                  size_t numRecords, int operation = REDUCE_SUM);

// PE optimized
void scanWithPE(cl::Buffer &d_input, const void *h_input,
                cl::Buffer &d_output, size_t numRecords,
                int operation = REDUCE_SUM);

// ============================================================================
// Reduce (Aggregation) Operation
// ============================================================================
// Aggregate the y component of records
// Operations: REDUCE_SUM, REDUCE_MAX, REDUCE_MIN, REDUCE_AVERAGE

// Baseline
int reduceBaseline(cl::Buffer &d_input, const void *h_input,
                   size_t numRecords, int operation);

// PE optimized
int reduceWithPE(cl::Buffer &d_input, const void *h_input,
                 size_t numRecords, int operation);

// ============================================================================
// Internal Helper Functions
// ============================================================================

// Get optimal work group size for a device
size_t getOptimalWorkGroupSize(const cl::Device &device, const cl::Kernel &kernel);

// Calculate global work size (must be multiple of local work size)
size_t roundUpToMultiple(size_t value, size_t multiple);

#endif // PRIMITIVES_H

