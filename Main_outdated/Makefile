# ============================================================================
# PE In-Cache Co-Processing Makefile
# ============================================================================

ifndef CPPC
	CPPC=g++
endif

# Compiler flags
CCFLAGS=-O2 -ffast-math -std=c++11 -Wall

# Libraries
LIBS = -lm -lOpenCL -lpthread

# Include directories
COMMON_DIR = .
INC = -I $(COMMON_DIR)

# Executable name
EXEC = main

# Source files
SRCS = main.cpp common.cpp prefetch.cpp primitives.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(EXEC)
	@echo "Build complete. Run with: ./$(EXEC)"

# Link executable
$(EXEC): $(OBJS)
	$(CPPC) $(OBJS) $(LIBS) -o $(EXEC)

# Compile source files
%.o: %.cpp
	$(CPPC) -c $< $(INC) $(CCFLAGS) -o $@

# Dependencies
main.o: main.cpp common.h prefetch.h primitives.h
common.o: common.cpp common.h OpenCL/device_picker.hpp OpenCL/err_code.h
prefetch.o: prefetch.cpp prefetch.h common.h
primitives.o: primitives.cpp primitives.h common.h prefetch.h

# Clean build artifacts
clean:
	rm -f $(EXEC) $(OBJS)

# Run the program
run: $(EXEC)
	./$(EXEC)

# Run with kernel directory
run-debug: $(EXEC)
	./$(EXEC) ./

# Help
help:
	@echo "PE In-Cache Co-Processing Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the executable (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  run       - Build and run"
	@echo "  run-debug - Build and run with current directory for kernels"
	@echo "  help      - Show this help message"

.PHONY: all clean run run-debug help
