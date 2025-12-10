#include "common.h"

void EngineStart() {
    cl_init(CL_DEVICE_TYPE_CPU);
    cl_init(CL_DEVICE_TYPE_GPU);
    
}

int main() {
    EngineStart();
}