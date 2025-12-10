#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include "OpenCL/cl.hpp"
#include <CL/cl.h>

void cl_init(cl_device_type TYPE);
void cl_init_common();