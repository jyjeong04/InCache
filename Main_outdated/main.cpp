#include "OpenCL/err_code.h"
#include "common.h"
#include "prefetch.h"
#include "primitives.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>

// ============================================================================
// Test Configuration
// ============================================================================
#define TEST_SIZE (16 * 1024 * 1024) // 16M records = 128MB data
#define FILTER_SELECTIVITY 0.1f      // 10% selectivity
#define NUM_ITERATIONS 10
#define WARMUP_ITERATIONS 3

char *programDir = (char *)"";

// ============================================================================
// Test Data Generation
// ============================================================================
void generateTestData(Record *data, size_t numRecords, int maxValue) {
  srand(42); // Fixed seed for reproducibility
  for (size_t i = 0; i < numRecords; i++) {
    data[i].s[0] = i;                 // x: record ID
    data[i].s[1] = rand() % maxValue; // y: key value
  }
}

void generateSequentialData(Record *data, size_t numRecords) {
  for (size_t i = 0; i < numRecords; i++) {
    data[i].s[0] = i;
    data[i].s[1] = i;
  }
}

// ============================================================================
// Benchmark Utilities
// ============================================================================
double getTimeMs() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(duration)
             .count() /
         1000.0;
}

void printResults(const char *testName, double timeMs, size_t dataSize) {
  double throughput =
      (dataSize / (1024.0 * 1024.0 * 1024.0)) / (timeMs / 1000.0);
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "[" << testName << "] Time: " << timeMs << " ms, "
            << "Throughput: " << throughput << " GB/s\n"
            << std::flush;
}

// ============================================================================
// Test Functions
// ============================================================================

// Test 1: Filter operation with PE optimization
void testFilter(size_t numRecords, bool usePE) {
  std::cout << "\n=== Filter Test (PE=" << (usePE ? "ON" : "OFF")
            << ", Records=" << numRecords << ") ===\n"
            << std::flush;

  size_t dataSize = numRecords * sizeof(Record);

  // Allocate host memory
  Record *h_input = (Record *)host_malloc(dataSize);
  Record *h_output = (Record *)host_malloc(dataSize);

  if (!h_input || !h_output) {
    std::cerr << "Failed to allocate host memory\n";
    return;
  }

  // Generate test data

  int maxValue = 1000;
  generateTestData(h_input, numRecords, maxValue);

  // Filter condition: key >= 0 && key <= maxValue * FILTER_SELECTIVITY
  int smallKey = 0;
  int largeKey = (int)(maxValue * FILTER_SELECTIVITY);

  // Allocate device buffers with CL_MEM_ALLOC_HOST_PTR for APU zero-copy
  cl::Buffer d_input = cl_malloc(dataSize, CL_MEM_READ_ONLY);
  cl::Buffer d_output = cl_malloc(dataSize, CL_MEM_WRITE_ONLY);

  // Warm-up iterations
  for (int i = 0; i < WARMUP_ITERATIONS; i++) {

    if (usePE) {
      filterWithPE(d_input, h_input, d_output, numRecords, smallKey, largeKey);
    } else {
      filterBaseline(d_input, h_input, d_output, numRecords, smallKey,
                     largeKey);
    }
  }

  // Benchmark
  std::vector<double> times;
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    double start = getTimeMs();

    size_t outputSize;
    if (usePE) {
      outputSize = filterWithPE(d_input, h_input, d_output, numRecords,
                                smallKey, largeKey);
    } else {
      outputSize = filterBaseline(d_input, h_input, d_output, numRecords,
                                  smallKey, largeKey);
    }

    double end = getTimeMs();
    times.push_back(end - start);

    if (i == 0) {
      std::cout << "Output size: " << outputSize << " records ("
                << (100.0 * outputSize / numRecords) << "%)\n";
    }
  }

  // Calculate statistics (exclude outliers)
  std::sort(times.begin(), times.end());
  double avgTime = 0;
  int validCount = (int)times.size() - 4; // Exclude 2 highest and 2 lowest
  if (validCount > 0) {
    for (int i = 2; i < (int)times.size() - 2; i++) {
      avgTime += times[i];
    }
    avgTime /= validCount;
  } else {
    // Not enough samples, use all
    for (size_t i = 0; i < times.size(); i++) {
      avgTime += times[i];
    }
    avgTime /= times.size();
  }

  printResults(usePE ? "Filter-PE" : "Filter-Baseline", avgTime, dataSize);

  // Cleanup
  host_free(h_input);
  host_free(h_output);
}

// Test 2: Map operation
void testMap(size_t numRecords, bool usePE) {
  std::cout << "\n=== Map Test (PE=" << (usePE ? "ON" : "OFF")
            << ", Records=" << numRecords << ") ===\n";

  size_t dataSize = numRecords * sizeof(Record);

  Record *h_input = (Record *)host_malloc(dataSize);
  int *h_output1 = (int *)host_malloc(numRecords * sizeof(int));
  int *h_output2 = (int *)host_malloc(numRecords * sizeof(int));

  if (!h_input || !h_output1 || !h_output2) {
    std::cerr << "Failed to allocate host memory\n";
    return;
  }

  generateTestData(h_input, numRecords, 1000);

  // Allocate device buffers
  cl::Buffer d_input = cl_malloc(dataSize, CL_MEM_READ_ONLY);
  cl::Buffer d_output1 = cl_malloc(numRecords * sizeof(int), CL_MEM_WRITE_ONLY);
  cl::Buffer d_output2 = cl_malloc(numRecords * sizeof(int), CL_MEM_WRITE_ONLY);

  // Warm-up
  for (int i = 0; i < WARMUP_ITERATIONS; i++) {
    if (usePE) {
      mapWithPE(d_input, h_input, d_output1, d_output2, numRecords);
    } else {
      mapBaseline(d_input, h_input, d_output1, d_output2, numRecords);
    }
  }

  // Benchmark
  std::vector<double> times;
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    double start = getTimeMs();

    if (usePE) {
      mapWithPE(d_input, h_input, d_output1, d_output2, numRecords);
    } else {
      mapBaseline(d_input, h_input, d_output1, d_output2, numRecords);
    }

    double end = getTimeMs();
    times.push_back(end - start);
  }

  std::sort(times.begin(), times.end());
  double avgTime = 0;
  int validCount = times.size() - 4;
  for (int i = 2; i < (int)times.size() - 2; i++) {
    avgTime += times[i];
  }
  avgTime /= validCount;

  printResults(usePE ? "Map-PE" : "Map-Baseline", avgTime, dataSize);

  host_free(h_input);
  host_free(h_output1);
  host_free(h_output2);
}

// Test 3: Reduce operation
void testReduce(size_t numRecords, int operation, bool usePE) {
  const char *opNames[] = {"SUM", "MAX", "MIN", "AVG"};
  std::cout << "\n=== Reduce Test (" << opNames[operation]
            << ", PE=" << (usePE ? "ON" : "OFF") << ", Records=" << numRecords
            << ") ===\n";

  size_t dataSize = numRecords * sizeof(Record);

  Record *h_input = (Record *)host_malloc(dataSize);
  if (!h_input) {
    std::cerr << "Failed to allocate host memory\n";
    return;
  }

  generateTestData(h_input, numRecords, 1000);

  // Allocate device buffer
  cl::Buffer d_input = cl_malloc(dataSize, CL_MEM_READ_ONLY);

  // Warm-up
  for (int i = 0; i < WARMUP_ITERATIONS; i++) {
    if (usePE) {
      reduceWithPE(d_input, h_input, numRecords, operation);
    } else {
      reduceBaseline(d_input, h_input, numRecords, operation);
    }
  }

  // Benchmark
  std::vector<double> times;
  int result = 0;
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    double start = getTimeMs();

    if (usePE) {
      result = reduceWithPE(d_input, h_input, numRecords, operation);
    } else {
      result = reduceBaseline(d_input, h_input, numRecords, operation);
    }

    double end = getTimeMs();
    times.push_back(end - start);

    if (i == 0) {
      std::cout << "Result: " << result << "\n";
    }
  }

  std::sort(times.begin(), times.end());
  double avgTime = 0;
  int validCount = times.size() - 4;
  for (int i = 2; i < (int)times.size() - 2; i++) {
    avgTime += times[i];
  }
  avgTime /= validCount;

  printResults(usePE ? "Reduce-PE" : "Reduce-Baseline", avgTime, dataSize);

  host_free(h_input);
}

// ============================================================================
// Engine Management
// ============================================================================
void EngineStart() {

  std::cout << "========================================\n";
  std::cout << "  PE In-Cache Co-Processing Engine\n";
  std::cout << "========================================\n";
  std::cout << std::flush;

  cl_init();

  cl_init_common();

  cl_prepareProgram("primitive.cl", programDir);

  // Initialize prefetch system
  prefetchInit();

  // Initialize primitives
  primitivesInit();

  std::cout << "\n[Engine] Initialization complete!\n" << std::flush;
}

void EngineStop() {
  std::cout << "\n[Engine] Shutting down...\n";
  primitivesCleanup();
  prefetchCleanup();
}

// ============================================================================
// Main Function
// ============================================================================
int main(int argc, char **argv) {

  // Parse command line arguments
  if (argc >= 2) {
    programDir = argv[1];
  }

  try {

    EngineStart();

    // Run benchmarks with different data sizes
    std::vector<size_t> testSizes = {
        1 * 1024 * 1024,  // 1M records (8MB)
        4 * 1024 * 1024,  // 4M records (32MB)
        16 * 1024 * 1024, // 16M records (128MB)
    };

    for (size_t size : testSizes) {

      std::cout << "\n****************************************\n";
      std::cout << "Testing with " << (size / (1024 * 1024)) << "M records ("
                << (size * sizeof(Record) / (1024 * 1024)) << " MB)\n";
      std::cout << "****************************************\n";

      // Filter tests
      testFilter(size, false); // Baseline
      testFilter(size, true);  // With PE

      // Map tests
      testMap(size, false);
      testMap(size, true);

      // Reduce tests (SUM)
      testReduce(size, REDUCE_SUM, false);
      testReduce(size, REDUCE_SUM, true);
    }

    EngineStop();

  } catch (cl::Error &err) {
    std::cerr << "OpenCL Error: " << err.what() << " (" << err_code(err.err())
              << ")" << std::endl;
    return 1;
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    cl_cleanup();
    return 1;
  }

  cl_cleanup();

  // Use _Exit to avoid pocl driver crash during global destructor cleanup
  _Exit(0);
}
