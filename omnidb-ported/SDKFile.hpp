/**
 * SDKFile.hpp - Replacement for AMD APP SDK headers
 * Minimal implementation for file operations
 */

#ifndef SDK_FILE_HPP
#define SDK_FILE_HPP

#include <cstdlib>
#include <fstream>
#include <string>

namespace streamsdk {

class SDKFile {
public:
  static bool readBinaryFromFile(const char *filename, char **data,
                                 size_t *size) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
      return false;

    *size = file.tellg();
    file.seekg(0, std::ios::beg);

    *data = new char[*size];
    file.read(*data, *size);
    file.close();

    return true;
  }

  static bool writeBinaryToFile(const char *filename, const char *data,
                                size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
      return false;

    file.write(data, size);
    file.close();

    return true;
  }
};

} // namespace streamsdk

#endif // SDK_FILE_HPP
