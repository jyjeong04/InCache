/**
 * SDKCommon.hpp - Replacement for AMD APP SDK headers
 * This file provides a minimal implementation for Linux + Intel OpenCL
 * environment
 */

#ifndef SDK_COMMON_HPP
#define SDK_COMMON_HPP

#include <CL/cl.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

#define CHECK_OPENCL_ERROR(status, msg)                                        \
  if (status != CL_SUCCESS) {                                                  \
    std::cerr << "Error: " << msg << " Error code: " << status << std::endl;   \
    return SDK_FAILURE;                                                        \
  }

#define CHECK_ALLOCATION(ptr, msg)                                             \
  if (ptr == NULL) {                                                           \
    std::cerr << "Error: " << msg << std::endl;                                \
    return SDK_FAILURE;                                                        \
  }

namespace streamsdk {

/**
 * Timer structure for performance measurement
 */
struct Timer {
  long long _start;
  long long _clocks;
  long long _freq;
};

/**
 * Table structure for formatted output
 */
struct Table {
  std::string _dataItems;
  std::string _delim;
  int _columnWidth;
  int _numColumns;
};

/**
 * SDKDeviceInfo class - stores OpenCL device information
 */
class SDKDeviceInfo {
public:
  cl_device_type dType;
  cl_uint venderId;
  cl_uint maxComputeUnits;
  cl_uint maxWorkItemDims;
  size_t *maxWorkItemSizes;
  size_t maxWorkGroupSize;
  cl_uint preferredCharVecWidth;
  cl_uint preferredShortVecWidth;
  cl_uint preferredIntVecWidth;
  cl_uint preferredLongVecWidth;
  cl_uint preferredFloatVecWidth;
  cl_uint preferredDoubleVecWidth;
  cl_uint preferredHalfVecWidth;
  cl_uint nativeCharVecWidth;
  cl_uint nativeShortVecWidth;
  cl_uint nativeIntVecWidth;
  cl_uint nativeLongVecWidth;
  cl_uint nativeFloatVecWidth;
  cl_uint nativeDoubleVecWidth;
  cl_uint nativeHalfVecWidth;
  cl_uint maxClockFrequency;
  cl_uint addressBits;
  cl_ulong maxMemAllocSize;
  cl_bool imageSupport;
  cl_uint maxReadImageArgs;
  cl_uint maxWriteImageArgs;
  size_t image2dMaxWidth;
  size_t image2dMaxHeight;
  size_t image3dMaxWidth;
  size_t image3dMaxHeight;
  size_t image3dMaxDepth;
  cl_uint maxSamplers;
  size_t maxParameterSize;
  cl_uint memBaseAddressAlign;
  cl_uint minDataTypeAlignSize;
  cl_device_fp_config singleFpConfig;
  cl_device_fp_config doubleFpConfig;
  cl_device_mem_cache_type globleMemCacheType;
  cl_uint globalMemCachelineSize;
  cl_ulong globalMemCacheSize;
  cl_ulong globalMemSize;
  cl_ulong maxConstBufSize;
  cl_uint maxConstArgs;
  cl_device_local_mem_type localMemType;
  cl_ulong localMemSize;
  cl_bool errCorrectionSupport;
  cl_bool hostUnifiedMem;
  size_t timerResolution;
  cl_bool endianLittle;
  cl_bool available;
  cl_bool compilerAvailable;
  cl_device_exec_capabilities execCapabilities;
  cl_command_queue_properties queueProperties;
  cl_platform_id platform;
  char *name;
  char *vendorName;
  char *driverVersion;
  char *profileType;
  char *deviceVersion;
  char *openclCVersion;
  char *extensions;

  SDKDeviceInfo() {
    maxWorkItemSizes = NULL;
    name = NULL;
    vendorName = NULL;
    driverVersion = NULL;
    profileType = NULL;
    deviceVersion = NULL;
    openclCVersion = NULL;
    extensions = NULL;
  }

  ~SDKDeviceInfo() {
    delete[] maxWorkItemSizes;
    delete[] name;
    delete[] vendorName;
    delete[] driverVersion;
    delete[] profileType;
    delete[] deviceVersion;
    delete[] openclCVersion;
    delete[] extensions;
  }

  int setDeviceInfo(cl_device_id deviceId);

  template <typename T>
  int checkVal(T input, T reference, std::string message,
               bool isAPIerror = true) const;
};

/**
 * KernelWorkGroupInfo class - stores kernel work group information
 */
class KernelWorkGroupInfo {
public:
  size_t kernelWorkGroupSize;
  cl_ulong localMemoryUsed;
  size_t compileWorkGroupSize[3];

  KernelWorkGroupInfo() : kernelWorkGroupSize(0), localMemoryUsed(0) {
    compileWorkGroupSize[0] = 0;
    compileWorkGroupSize[1] = 0;
    compileWorkGroupSize[2] = 0;
  }

  int setKernelWorkGroupInfo(cl_kernel &kernel, cl_device_id &device);

  template <typename T>
  int checkVal(T input, T reference, std::string message,
               bool isAPIerror = true) const;
};

/**
 * SDKCommon class - main utility class
 */
class SDKCommon {
private:
  std::vector<Timer *> _timers;

public:
  SDKCommon();
  ~SDKCommon();

  // Timer functions
  int createTimer();
  int resetTimer(int handle);
  int startTimer(int handle);
  int stopTimer(int handle);
  double readTimer(int handle);

  // Utility functions
  std::string getPath();

  template <typename T>
  void printArray(std::string header, const T *data, const int width,
                  const int height) const;

  template <typename T>
  int fillRandom(T *arrayPtr, const int width, const int height,
                 const T rangeMin, const T rangeMax, unsigned int seed = 0);

  template <typename T>
  int fillPos(T *arrayPtr, const int width, const int height);

  template <typename T>
  int fillConstant(T *arrayPtr, const int width, const int height, const T val);

  template <typename T> T roundToPowerOf2(T val);

  template <typename T> int isPowerOf2(T val);

  template <typename T>
  int checkVal(T input, T reference, std::string message,
               bool isAPIerror = true);

  template <typename T>
  std::string toString(T t, std::ios_base &(*r)(std::ios_base &));

  int waitForEventAndRelease(cl_event *event);
  int displayDevices(cl_platform_id platform, cl_device_type deviceType);
  int displayPlatformAndDevices(cl_platform_id platform,
                                const cl_device_id *devices,
                                const int deviceCount);
  int validateDeviceId(int deviceId, int deviceCount);
  int getDevices(cl_context &context, cl_device_id **devices, cl_int deviceId,
                 bool deviceIdEnabled);
  int getPlatform(cl_platform_id &platform, int platformId,
                  bool platformIdEnabled);

  bool compare(const float *refData, const float *data, const int length,
               const float epsilon = 1e-6f);
  bool compare(const double *refData, const double *data, const int length,
               const double epsilon = 1e-6);

  size_t getLocalThreads(const size_t globalThreads,
                         const size_t maxWorkItemSize);

  void printTable(Table *t);
  int fileToString(std::string &fileName, std::string &str);

  void error(const char *errorMsg);
  void error(std::string errorMsg);
  void expectedError(const char *errorMsg);
  void expectedError(std::string errorMsg);
};

} // namespace streamsdk

#endif // SDK_COMMON_HPP
