#ifndef COMMON_H
#define COMMON_H

#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS

#include "OpenCL/cl.hpp"
#include <CL/cl.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Debug output macro - comment out to disable debug output
// #define PE_DEBUG
#ifdef PE_DEBUG
#define DEBUG_PRINT(x) std::cerr << x << std::flush
#else
#define DEBUG_PRINT(x)
#endif

// ============================================================================
// Data Types (compatible with omnidb)
// ============================================================================
typedef cl_uint2 Record;

// Reduce operations
#define REDUCE_SUM 0
#define REDUCE_MAX 1
#define REDUCE_MIN 2
#define REDUCE_AVERAGE 3

// ============================================================================
// PE (Prefetching + Execution) Configuration
// ============================================================================
struct PEConfig {
  size_t chunkSize;       // Size of each chunk for prefetching (bytes)
  size_t numChunks;       // Number of chunks for pipelining
  bool enablePrefetch;    // Enable/disable prefetching
  float cpuExecRatio;     // Ratio of work for CPU execution (0.0 - 1.0)
};

// Default configuration
extern PEConfig peConfig;

// ============================================================================
// Device Fission Structure
// ============================================================================
// CPU is split into:
//   - prefetchDevice: 1 core dedicated to prefetching
//   - execCpuDevices: remaining cores for execution
// GPU is used entirely for execution

extern cl::Context context;
extern std::vector<cl::Device> devices;

// Original devices
extern cl::Device CPU;
extern cl::Device GPU;

// Fissioned CPU sub-devices
extern cl::Device prefetchDevice;              // 1 core for prefetching
extern std::vector<cl::Device> execCpuDevices; // Remaining cores for execution

// Command Queues
extern cl::CommandQueue prefetchQueue;              // Queue for prefetching
extern std::vector<cl::CommandQueue> execCpuQueues; // Queues for CPU execution
extern cl::CommandQueue execGpuQueue;               // Queue for GPU execution

// Program
extern cl::Program program;

// ============================================================================
// Initialization Functions
// ============================================================================
void cl_init();
void cl_init_common();
void cl_prepareProgram(const char *cSourceFile, const char *dir);
void cl_cleanup(); // Cleanup OpenCL resources before exit

// ============================================================================
// Memory Allocation Helpers
// ============================================================================
cl::Buffer cl_malloc(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE);
// Zero-copy buffer using host pointer (optimal for APU)
cl::Buffer cl_malloc_use_host(void *hostPtr, size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE);
void cl_free(cl::Buffer &buf);

// Host memory allocation (page-aligned for zero-copy)
void *host_malloc(size_t size);
void host_free(void *ptr);

// ============================================================================
// Utility Functions
// ============================================================================
int convertToString(const char *filename, std::string &s);
std::string appendPath(const std::string &dir, const std::string &file);

// Get device info
cl_uint getCpuComputeUnits();
cl_ulong getDeviceCacheSize(const cl::Device &dev);
cl_ulong getDeviceGlobalMemSize(const cl::Device &dev);

#endif // COMMON_H
