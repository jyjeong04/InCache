#include "common.h"
#include "OpenCL/device_picker.hpp"
#include "OpenCL/err_code.h"
#include <cstdlib>

// ============================================================================
// Global Variable Definitions
// ============================================================================
cl::Context context;
std::vector<cl::Device> devices;

// Original devices
cl::Device CPU;
cl::Device GPU;

// Fissioned CPU sub-devices
cl::Device prefetchDevice;
std::vector<cl::Device> execCpuDevices;

// Command Queues
cl::CommandQueue prefetchQueue;
std::vector<cl::CommandQueue> execCpuQueues;
cl::CommandQueue execGpuQueue;

// Program
cl::Program program;

// PE Configuration with defaults
PEConfig peConfig = {
    .chunkSize = 2 * 1024 * 1024, // 2MB chunks (maximize LLC utilization)
    .numChunks = 4,               // Double buffering with extra
    .enablePrefetch = true,
    .cpuExecRatio = 0.3f // 30% CPU, 70% GPU for execution
};

// ============================================================================
// Device Discovery
// ============================================================================
static bool findDevice(cl::Device &dev, cl_device_type type) {
  for (size_t i = 0; i < devices.size(); i++) {
    if (devices[i].getInfo<CL_DEVICE_TYPE>() == type) {
      dev = devices[i];
      return true;
    }
  }
  return false;
}

void cl_init() {

  try {

    unsigned numDevices = getDeviceList(devices);

    if (numDevices == 0) {
      throw cl::Error(CL_DEVICE_NOT_FOUND, "No OpenCL devices found");
    }

    // Find CPU and GPU

    bool foundCPU = findDevice(CPU, CL_DEVICE_TYPE_CPU);

    bool foundGPU = findDevice(GPU, CL_DEVICE_TYPE_GPU);

    std::string name;
    getDeviceName(CPU, name);

    std::cout << "\n[PE Init] Found CPU: " << name << "\n" << std::flush;

    cl_uint cpuCUs = CPU.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    std::cout << "[PE Init] CPU Compute Units: " << cpuCUs << "\n"
              << std::flush;

    getDeviceName(GPU, name);

    std::cout << "[PE Init] Found GPU: " << name << "\n" << std::flush;

    cl_uint gpuCUs = GPU.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    std::cout << "[PE Init] GPU Compute Units: " << gpuCUs << "\n"
              << std::flush;

  } catch (cl::Error &err) {
    std::cerr << "ERROR in cl_init: " << err.what() << " ("
              << err_code(err.err()) << ")" << std::endl;
    throw;
  }
}

// ============================================================================
// Device Fission and Context Creation
// ============================================================================
void cl_init_common() {
  try {
    cl_uint numCUs = CPU.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

    // ========================================================================
    // PE Device Fission Strategy:
    // - 1 core for Prefetching
    // - (numCUs - 1) cores for Execution
    // ========================================================================

    if (numCUs < 2) {
      std::cerr << "[PE Init] Warning: CPU has only " << numCUs
                << " CU(s). Prefetching will share with execution.\n";

      // Fallback: use same device for both
      prefetchDevice = CPU;
      execCpuDevices.push_back(CPU);
    } else {
      // Create sub-devices using PARTITION_BY_COUNTS
      // First partition: 1 core for prefetch
      // Second partition: remaining cores for execution

      std::vector<cl_device_partition_property> props;
      props.push_back(CL_DEVICE_PARTITION_BY_COUNTS);
      props.push_back(1);          // 1 core for prefetch
      props.push_back(numCUs - 1); // Remaining for execution
      props.push_back(CL_DEVICE_PARTITION_BY_COUNTS_LIST_END);
      props.push_back(0);

      std::vector<cl::Device> subDevices;

      try {
        CPU.createSubDevices(props.data(), &subDevices);

        if (subDevices.size() >= 2) {
          prefetchDevice = subDevices[0];          // 1 core
          execCpuDevices.push_back(subDevices[1]); // Remaining cores

          std::cout << "[PE Init] Device Fission successful:\n";
          std::cout << "  - Prefetch device: 1 CU\n";
          std::cout << "  - Execution device: " << (numCUs - 1) << " CUs\n";
        } else {
          throw cl::Error(CL_DEVICE_PARTITION_FAILED,
                          "Insufficient sub-devices created");
        }
      } catch (cl::Error &err) {
        // Fallback to PARTITION_EQUALLY if BY_COUNTS not supported
        std::cout
            << "[PE Init] BY_COUNTS failed, trying EQUALLY partition...\n";

        cl_device_partition_property equalProps[] = {
            CL_DEVICE_PARTITION_EQUALLY,
            1, // 1 CU per sub-device
            0};

        subDevices.clear();
        CPU.createSubDevices(equalProps, &subDevices);

        if (subDevices.size() >= 2) {
          prefetchDevice = subDevices[0]; // First sub-device for prefetch

          // Remaining sub-devices for execution
          for (size_t i = 1; i < subDevices.size(); i++) {
            execCpuDevices.push_back(subDevices[i]);
          }

          std::cout << "[PE Init] Device Fission (EQUALLY) successful:\n";
          std::cout << "  - Prefetch device: 1 sub-device\n";
          std::cout << "  - Execution devices: " << execCpuDevices.size()
                    << " sub-devices\n";
        } else {
          // Ultimate fallback: use CPU for both
          prefetchDevice = CPU;
          execCpuDevices.push_back(CPU);
          std::cout << "[PE Init] Fallback: using same CPU for prefetch and "
                       "execution\n";
        }
      }
    }

    // ========================================================================
    // Create Context with all devices
    // ========================================================================
    std::vector<cl::Device> allDevices;
    allDevices.push_back(prefetchDevice);
    for (auto &dev : execCpuDevices) {
      allDevices.push_back(dev);
    }
    allDevices.push_back(GPU);

    context = cl::Context(allDevices);
    std::cout << "[PE Init] Context created with " << allDevices.size()
              << " devices\n";

    // ========================================================================
    // Create Command Queues
    // ========================================================================

    // Prefetch queue (in-order, for sequential prefetching)
    prefetchQueue =
        cl::CommandQueue(context, prefetchDevice, CL_QUEUE_PROFILING_ENABLE);
    std::cout << "[PE Init] Prefetch queue created\n";

    // CPU execution queues
    for (size_t i = 0; i < execCpuDevices.size(); i++) {
      execCpuQueues.emplace_back(context, execCpuDevices[i],
                                 CL_QUEUE_PROFILING_ENABLE);
    }
    std::cout << "[PE Init] " << execCpuQueues.size()
              << " CPU execution queue(s) created\n";

    // GPU execution queue
    execGpuQueue = cl::CommandQueue(context, GPU, CL_QUEUE_PROFILING_ENABLE);
    std::cout << "[PE Init] GPU execution queue created\n";

    // Print cache info
    cl_ulong cpuCache = getDeviceCacheSize(CPU);
    cl_ulong gpuCache = getDeviceCacheSize(GPU);
    std::cout << "[PE Init] CPU Global Cache: " << cpuCache / 1024 << " KB\n";
    std::cout << "[PE Init] GPU Global Cache: " << gpuCache / 1024 << " KB\n";

  } catch (cl::Error &err) {
    std::cerr << "ERROR in cl_init_common: " << err.what() << " ("
              << err_code(err.err()) << ")" << std::endl;
    throw;
  }
}

// ============================================================================
// Program Compilation
// ============================================================================
void cl_prepareProgram(const char *cSourceFile, const char *dir) {
  try {
    std::string fullPath = appendPath(dir, cSourceFile);
    std::cout << "[PE Init] Loading kernel: " << fullPath << "\n";

    std::string sourceStr;
    if (convertToString(fullPath.c_str(), sourceStr) != 0) {
      throw cl::Error(CL_INVALID_PROGRAM, "Failed to read kernel file");
    }

    cl::Program::Sources sources;
    sources.push_back({sourceStr.c_str(), sourceStr.length()});
    program = cl::Program(context, sources);

    // Build for all devices in context
    try {
      program.build("-cl-fast-relaxed-math");
      std::cout << "[PE Init] Kernel compilation successful!\n";
    } catch (cl::Error &err) {
      // Print build log on failure
      std::cerr << "Build failed. Build log:\n";

      std::vector<cl::Device> contextDevices =
          context.getInfo<CL_CONTEXT_DEVICES>();
      for (auto &dev : contextDevices) {
        std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
        if (!log.empty()) {
          std::string devName;
          getDeviceName(dev, devName);
          std::cerr << "=== " << devName << " ===\n" << log << "\n";
        }
      }
      throw;
    }
  } catch (cl::Error &err) {
    std::cerr << "ERROR in cl_prepareProgram: " << err.what() << " ("
              << err_code(err.err()) << ")" << std::endl;
    throw;
  }
}

// ============================================================================
// Memory Allocation
// ============================================================================
cl::Buffer cl_malloc(size_t size, cl_mem_flags flags) {
  cl_int err;
  // Use CL_MEM_ALLOC_HOST_PTR for APU zero-copy optimization
  cl::Buffer buf(context, flags | CL_MEM_ALLOC_HOST_PTR, size, nullptr, &err);
  if (err != CL_SUCCESS) {
    throw cl::Error(err, "cl_malloc failed");
  }
  return buf;
}

cl::Buffer cl_malloc_use_host(void *hostPtr, size_t size, cl_mem_flags flags) {
  cl_int err;
  // CL_MEM_USE_HOST_PTR: Use the provided host pointer directly
  // This enables true zero-copy on APU - no data movement needed
  // The host pointer MUST be page-aligned (use host_malloc)
  cl::Buffer buf(context, flags | CL_MEM_USE_HOST_PTR, size, hostPtr, &err);
  if (err != CL_SUCCESS) {
    throw cl::Error(err, "cl_malloc_use_host failed");
  }
  return buf;
}

void cl_free(cl::Buffer &buf) {
  // Buffer destructor handles cleanup
  buf = cl::Buffer();
}

void *host_malloc(size_t size) {
  // Allocate page-aligned memory for efficient DMA transfers
  void *ptr = nullptr;
#ifdef _WIN32
  ptr = _aligned_malloc(size, 4096);
#else
  if (posix_memalign(&ptr, 4096, size) != 0) {
    ptr = nullptr;
  }
#endif
  return ptr;
}

void host_free(void *ptr) {
  if (ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }
}

void cl_cleanup() {
  // Explicitly release OpenCL resources in the correct order
  // to avoid pocl driver crashes at exit

  // Finish all queues first
  try {
    prefetchQueue.finish();
    for (auto &q : execCpuQueues) {
      q.finish();
    }
    execGpuQueue.finish();
  } catch (...) {
  }

  // Clear command queues
  prefetchQueue = cl::CommandQueue();
  execCpuQueues.clear();
  execGpuQueue = cl::CommandQueue();

  // Release program (this is what was causing the crash)
  program = cl::Program();

  // Clear device lists
  execCpuDevices.clear();

  // Clear context last
  context = cl::Context();
}

// ============================================================================
// Utility Functions
// ============================================================================
int convertToString(const char *filename, std::string &s) {
  std::fstream f(filename, std::fstream::in | std::fstream::binary);

  if (!f.is_open()) {
    std::cerr << "Error: Failed to open file " << filename << "\n";
    return 1;
  }

  f.seekg(0, std::fstream::end);
  size_t size = static_cast<size_t>(f.tellg());
  f.seekg(0, std::fstream::beg);

  std::vector<char> buffer(size + 1);
  f.read(buffer.data(), size);
  f.close();

  buffer[size] = '\0';
  s = buffer.data();
  return 0;
}

std::string appendPath(const std::string &dir, const std::string &file) {
  if (dir.empty()) {
    return file;
  }
  if (dir.back() == '/' || dir.back() == '\\') {
    return dir + file;
  }
  return dir + "/" + file;
}

cl_uint getCpuComputeUnits() {
  return CPU.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
}

cl_ulong getDeviceCacheSize(const cl::Device &dev) {
  return dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
}

cl_ulong getDeviceGlobalMemSize(const cl::Device &dev) {
  return dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
}
