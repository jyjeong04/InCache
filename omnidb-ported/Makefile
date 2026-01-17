# Makefile for OmniDB OpenCL on Linux + Intel
# Target: Linux with Intel CPU/GPU using PoCL (Portable Computing Language)

# Compiler and flags
CXX = g++
CC = gcc
CXXFLAGS = -std=c++11 -O2 -Wall -fPIC -D__linux__
CFLAGS = -O2 -Wall -fPIC -D__linux__

# OpenCL configuration
# Use pkg-config if available, otherwise use standard paths
OPENCL_CFLAGS = $(shell pkg-config --cflags OpenCL 2>/dev/null || echo "-I/usr/include")
OPENCL_LIBS = $(shell pkg-config --libs OpenCL 2>/dev/null || echo "-lOpenCL")

# Include paths
INCLUDES = -I. $(OPENCL_CFLAGS)

# Libraries
LIBS = $(OPENCL_LIBS) -lpthread -lm

# Build directory
BUILD_DIR = build

# Source files
SOURCES = \
	mainProgram.cpp \
	common.cpp \
	SDKCommmon.cpp \
	Handshake.cpp \
	Helper.cpp \
	scheduler.cpp \
	KernelScheduler.cpp \
	CSSTree.cpp \
	PrimitiveCommon.cpp \
	Selection.cpp \
	Projection.cpp \
	GroupBy.cpp \
	AggAfterGroupBy.cpp \
	Sort.cpp \
	BinaryJoin.cpp \
	IndexJoin.cpp \
	Singular.cpp \
	SharedFunction.cpp \
	generator.cpp \
	spinlock.cpp \
	MidNumber.cpp \
	Validate.cpp

# Test sources (can be built separately)
TEST_SOURCES = \
	testGroupBy.cpp \
	testFilter.cpp \
	testMap.cpp \
	testProjection.cpp \
	testReduce.cpp \
	testRIDList.cpp \
	testScan.cpp \
	testScatter.cpp \
	testSort.cpp \
	testSplit.cpp \
	testValue.cpp \
	testAggAfterGB.cpp \
	testHJ.cpp \
	testINLJ.cpp \
	testNinlj.cpp \
	testSMJ.cpp

# Object files (in build directory)
OBJECTS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
TEST_OBJECTS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(TEST_SOURCES))

# Executable name
TARGET = exec

# Shared library name (optional)
SHARED_LIB = libomnidb_opencl.so

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)
	@echo "Build complete: $(TARGET)"

# Build shared library
shared: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -shared -o $(SHARED_LIB) $^ $(LIBS)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Compile C++ source files
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET) $(SHARED_LIB)
	@echo "Clean complete"

# Install (optional)
install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/
	install -m 644 primitive.cl /usr/local/share/omnidb/

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Check OpenCL environment
check-opencl:
	@echo "Checking OpenCL installation..."
	@clinfo -l || echo "clinfo not found, please install clinfo package"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the executable (default)"
	@echo "  shared       - Build shared library"
	@echo "  clean        - Remove build artifacts"
	@echo "  install      - Install to system"
	@echo "  debug        - Build with debug symbols"
	@echo "  check-opencl - Check OpenCL installation"
	@echo "  help         - Show this message"
	@echo ""
	@echo "Requirements:"
	@echo "  - OpenCL development headers (ocl-icd-opencl-dev or similar)"
	@echo "  - PoCL (Portable Computing Language) for CPU+GPU support"
	@echo "  - Intel OpenCL runtime for Intel GPU support"

.PHONY: all clean install debug check-opencl help shared

