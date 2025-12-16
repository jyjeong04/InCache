#ifndef PREFETCH_H
#define PREFETCH_H

#include "common.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>

// ============================================================================
// Prefetch Configuration
// ============================================================================
#define PREFETCH_CHUNK_SIZE (64 * 1024)    // 64KB per chunk
#define PREFETCH_BUFFER_COUNT 4             // Number of prefetch buffers
#define PREFETCH_LOOKAHEAD 2                // Chunks to prefetch ahead

// ============================================================================
// Prefetch Buffer Structure
// ============================================================================
struct PrefetchBuffer {
  void *hostPtr;           // Host pointer (pinned memory)
  cl::Buffer deviceBuf;    // Device buffer
  size_t size;             // Current data size
  size_t offset;           // Offset in source data
  cl::Event mapEvent;      // Event for map completion
  cl::Event unmapEvent;    // Event for unmap completion
  bool ready;              // Data is ready for GPU
  bool inUse;              // Buffer is being used
};

// ============================================================================
// Prefetch Manager Class
// ============================================================================
class PrefetchManager {
public:
  PrefetchManager();
  ~PrefetchManager();

  // Initialize prefetch system
  void init(size_t chunkSize = PREFETCH_CHUNK_SIZE, 
            size_t bufferCount = PREFETCH_BUFFER_COUNT);
  
  // Cleanup
  void cleanup();

  // Start prefetching data from host to device
  // Returns the number of chunks to process
  size_t startPrefetch(cl::Buffer &deviceBuf, const void *hostData, 
                       size_t totalSize, size_t elementSize);

  // Get next ready chunk for processing
  // Returns chunk index, or -1 if no chunk ready
  int getReadyChunk(size_t &offset, size_t &size);

  // Mark chunk as processed
  void markChunkProcessed(int chunkIndex);

  // Wait for all prefetch operations to complete
  void waitAll();

  // Check if prefetching is complete
  bool isComplete() const;

  // Get chunk size
  size_t getChunkSize() const { return m_chunkSize; }

private:
  // Prefetch worker function (runs on prefetch CPU core)
  void prefetchWorker();

  // Allocate pinned buffers
  void allocateBuffers();
  void freeBuffers();

private:
  size_t m_chunkSize;
  size_t m_bufferCount;
  
  std::vector<PrefetchBuffer> m_buffers;
  
  // Current prefetch state
  const void *m_sourceData;
  cl::Buffer *m_targetBuffer;
  size_t m_totalSize;
  size_t m_elementSize;
  size_t m_currentOffset;
  size_t m_totalChunks;
  size_t m_prefetchedChunks;
  size_t m_processedChunks;

  // Synchronization
  std::mutex m_mutex;
  std::condition_variable m_cv;
  std::atomic<bool> m_running;
  std::atomic<bool> m_complete;

  bool m_initialized;
};

// ============================================================================
// Global Prefetch Manager
// ============================================================================
extern PrefetchManager gPrefetchManager;

// ============================================================================
// Prefetch API Functions
// ============================================================================

// Initialize prefetch system
void prefetchInit();

// Cleanup prefetch system
void prefetchCleanup();

// Prefetch data using CPU core while GPU processes
// This function uses the prefetch CPU core to load data into LLC
// while GPU execution proceeds on cached data
void prefetchToDevice(cl::Buffer &deviceBuf, const void *hostData, 
                      size_t size, cl::Event *completeEvent = nullptr);

// Prefetch with chunk callback
// Callback is invoked for each ready chunk with (offset, size, chunkIndex)
typedef void (*PrefetchCallback)(size_t offset, size_t size, int chunkIndex, void *userData);

void prefetchWithCallback(cl::Buffer &deviceBuf, const void *hostData,
                          size_t size, PrefetchCallback callback, void *userData);

// Simple synchronous prefetch (for baseline comparison)
void prefetchSync(cl::Buffer &deviceBuf, const void *hostData, size_t size);

#endif // PREFETCH_H

