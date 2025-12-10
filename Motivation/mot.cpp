#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#define TESTSIZE 8

#include "OpenCL/cl.hpp"
#include "OpenCL/device_picker.hpp"
#include "OpenCL/err_code.h"
#include "OpenCL/util.hpp"
#include "OpenCL/wtime.c"
#include <iomanip>
#include <iostream>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <vector>

int main(int argc, char *argv[]) {
  int arrsize = TESTSIZE * 1024 * 1024 / sizeof(uint);
  std::vector<uint> arr(arrsize);
  for (int i = 0; i < arrsize; i++) {
    arr[i] = i;
  }

  try {
    cl_uint deviceIndex = 0;

    // Simple argument parsing: ./hj 1 or --device 1
    if (argc >= 2) {
      if (isdigit(argv[1][0])) {
        deviceIndex = atoi(argv[1]);
      } else {
        parseArguments(argc, argv, &deviceIndex);
      }
    }
    std::vector<cl::Device> devices;
    unsigned numDevices = getDeviceList(devices);

    if (numDevices == 0) {
      std::cerr << "ERROR: No OpenCL devices found.\n";
      return 1;
    }

    if (deviceIndex >= numDevices) {
      std::cerr << "ERROR: Device index " << deviceIndex
                << " is out of range. Available devices: 0-" << (numDevices - 1)
                << "\n";
      return 1;
    }

    cl::Device device = devices[deviceIndex];

    std::string name;
    getDeviceName(device, name);
    std::cout << "\nUsing OpenCL Device: " << name << "\n";

    std::vector<cl::Device> chosen_device;
    chosen_device.push_back(device);
    cl::Context context(chosen_device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Create programs and kernels
    cl::Program program(context, util::loadProgram("mot.cl"), true);

    cl::make_kernel<cl::Buffer> mot(program, "mot");
    cl::Buffer arr_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                       sizeof(uint) * arrsize, arr.data());

    // Get device cache size to determine if dataset fits in cache
    cl_ulong cache_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
    size_t dataset_size_bytes = TESTSIZE * 1024 * 1024;

    std::cout << "Device cache size: " << cache_size / (1024) << " MB\n";
    std::cout << "Dataset size: " << TESTSIZE << " MB\n";

    bool is_in_cache = (dataset_size_bytes <= cache_size);
    std::cout << "Cache status: " << (is_in_cache ? "IN-CACHE" : "OUT-OF-CACHE")
              << "\n";

    const int NUM_ITERATIONS = 100;   // Number of iterations for statistics
    const int WARMUP_ITERATIONS = 10; // Warm-up iterations (discarded)

    std::vector<double> kernel_times;
    kernel_times.reserve(NUM_ITERATIONS);

    // ========== MEASUREMENT ==========
    std::string cache_status = is_in_cache ? "IN-CACHE" : "OUT-OF-CACHE";
    std::cout << "\n=== MEASUREMENT ===\n";

    // Warm-up: run multiple times
    // for (int i = 0; i < WARMUP_ITERATIONS; i++) {
    //   cl::Event warmup_event =
    //       mot(cl::EnqueueArgs(queue, cl::NDRange(arrsize)), arr_buf);
    //   queue.finish();
    // }

    std::cout << "Measuring " << NUM_ITERATIONS << " iterations...\n";

    // Measure performance
    for (int i = 0; i < NUM_ITERATIONS; i++) {
      cl::Event kernel_event =
          mot(cl::EnqueueArgs(queue, cl::NDRange(arrsize)), arr_buf);
      queue.finish();

      cl_ulong start_time =
          kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      cl_ulong end_time =
          kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
      cl_ulong kernel_time_ns = end_time - start_time;

      kernel_times.push_back(kernel_time_ns / 1e9);
    }

    // Calculate statistics: exclude top 10 and bottom 10, then calculate mean
    std::sort(kernel_times.begin(), kernel_times.end());

    const int EXCLUDE_COUNT = 10;
    const int VALID_COUNT = NUM_ITERATIONS - 2 * EXCLUDE_COUNT;

    double kernel_mean = 0.0;
    // Skip bottom 10 and top 10
    for (int i = EXCLUDE_COUNT; i < NUM_ITERATIONS - EXCLUDE_COUNT; i++) {
      kernel_mean += kernel_times[i];
    }
    kernel_mean /= VALID_COUNT;

    double data_size_gb = TESTSIZE / 1024.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nResults: " << kernel_mean << " s\n"
              << TESTSIZE << "MB Throughput: " << data_size_gb / kernel_mean
              << " GB/s\n";

  } catch (cl::Error err) {
    std::cout << "Exception\n";
    std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")"
              << std::endl;
  }

  //===================== OpenCL Join End ==========================

  return 0;
}
