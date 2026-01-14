/**
 * SDKApplication.hpp - Replacement for AMD APP SDK headers
 * Minimal implementation for application framework
 */

#ifndef SDK_APPLICATION_HPP
#define SDK_APPLICATION_HPP

#include <iostream>
#include <string>

namespace streamsdk {

class SDKApplication {
public:
  SDKApplication() {}
  virtual ~SDKApplication() {}

  virtual int initialize() { return 0; }
  virtual int setup() { return 0; }
  virtual int run() { return 0; }
  virtual int verifyResults() { return 0; }
  virtual int cleanup() { return 0; }
};

} // namespace streamsdk

#endif // SDK_APPLICATION_HPP
