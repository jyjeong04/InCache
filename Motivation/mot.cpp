#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS

#include "../OpenCL/cl.hpp"
#include "../OpenCL/device_picker.hpp"
#include "../OpenCL/err_code.h"
#include "../OpenCL/util.hpp"
#include "../OpenCL/wtime.c"
#include <iostream>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <vector>

int main(int argc, char *argv[]) {

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

    cl::make_kernel<cl::Buffer, cl::Buffer> b1(program, "b1");
    cl::make_kernel<cl::Buffer, cl::Buffer> b2(program, "b2");
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> b3(program,
                                                                       "b3");
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> b4(program,
                                                                       "b4");
    cl::make_kernel<cl::Buffer, cl::Buffer> p1(program, "p1");
    cl::make_kernel<cl::Buffer, cl::Buffer> p2(program, "p2");
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
        p3(program, "p3");
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
        p4(program, "p4");

  } catch (cl::Error err) {
    std::cout << "Exception\n";
    std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")"
              << std::endl;
  }

  //===================== OpenCL Join End ==========================

  return 0;
}
