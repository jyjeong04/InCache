#include "prefetch.h"
#include <cstring>

// ============================================================================
// Global Prefetch Manager Instance
// ============================================================================
PrefetchManager gPrefetchManager;

// ============================================================================
// PrefetchManager Implementation
// ============================================================================
PrefetchManager::PrefetchManager()
    : m_chunkSize(PREFETCH_CHUNK_SIZE),
      m_bufferCount(PREFETCH_BUFFER_COUNT),
      m_sourceData(nullptr),
      m_targetBuffer(nullptr),
      m_totalSize(0),
      m_elementSize(0),
      m_currentOffset(0),
      m_totalChunks(0),
      m_prefetchedChunks(0),
      m_processedChunks(0),
      m_running(false),
      m_complete(true),
      m_initialized(false) {}

PrefetchManager::~PrefetchManager() { cleanup(); }

void PrefetchManager::init(size_t chunkSize, size_t bufferCount) {
  if (m_initialized) {
    return;
  }

  m_chunkSize = chunkSize;
  m_bufferCount = bufferCount;

  allocateBuffers();

  m_initialized = true;
  std::cout << "[Prefetch] Initialized with " << m_bufferCount << " buffers, "
            << m_chunkSize / 1024 << " KB chunks\n";
}

void PrefetchManager::cleanup() {
  if (!m_initialized) {
    return;
  }

  waitAll();
  freeBuffers();
  m_initialized = false;
}

void PrefetchManager::allocateBuffers() {
  m_buffers.resize(m_bufferCount);

  for (size_t i = 0; i < m_bufferCount; i++) {
    // Allocate pinned host memory for zero-copy
    m_buffers[i].hostPtr = host_malloc(m_chunkSize);
    if (!m_buffers[i].hostPtr) {
      throw std::runtime_error("Failed to allocate pinned memory for prefetch");
    }

    // Create device buffer with USE_HOST_PTR for zero-copy on APU
    m_buffers[i].deviceBuf =
        cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                   m_chunkSize);

    m_buffers[i].size = 0;
    m_buffers[i].offset = 0;
    m_buffers[i].ready = false;
    m_buffers[i].inUse = false;
  }
}

void PrefetchManager::freeBuffers() {
  for (auto &buf : m_buffers) {
    if (buf.hostPtr) {
      host_free(buf.hostPtr);
      buf.hostPtr = nullptr;
    }
    buf.deviceBuf = cl::Buffer();
  }
  m_buffers.clear();
}

size_t PrefetchManager::startPrefetch(cl::Buffer &deviceBuf,
                                      const void *hostData, size_t totalSize,
                                      size_t elementSize) {
  std::unique_lock<std::mutex> lock(m_mutex);

  m_sourceData = hostData;
  m_targetBuffer = &deviceBuf;
  m_totalSize = totalSize;
  m_elementSize = elementSize;
  m_currentOffset = 0;
  m_totalChunks = (totalSize + m_chunkSize - 1) / m_chunkSize;
  m_prefetchedChunks = 0;
  m_processedChunks = 0;
  m_complete = false;
  m_running = true;

  // Reset all buffers
  for (auto &buf : m_buffers) {
    buf.ready = false;
    buf.inUse = false;
  }

  // Start prefetching first few chunks using the prefetch CPU core
  for (size_t i = 0; i < std::min(m_bufferCount, m_totalChunks); i++) {
    prefetchWorker();
  }

  return m_totalChunks;
}

void PrefetchManager::prefetchWorker() {
  if (m_currentOffset >= m_totalSize) {
    return;
  }

  // Find a free buffer
  int bufIdx = -1;
  for (size_t i = 0; i < m_buffers.size(); i++) {
    if (!m_buffers[i].inUse) {
      bufIdx = i;
      break;
    }
  }

  if (bufIdx < 0) {
    return; // No free buffer
  }

  PrefetchBuffer &buf = m_buffers[bufIdx];
  buf.inUse = true;
  buf.offset = m_currentOffset;

  // Calculate chunk size
  size_t remaining = m_totalSize - m_currentOffset;
  buf.size = std::min(m_chunkSize, remaining);

  // Copy data to pinned buffer
  const char *src = static_cast<const char *>(m_sourceData) + m_currentOffset;
  std::memcpy(buf.hostPtr, src, buf.size);

  // Use clEnqueueMapBuffer to bring data into cache
  // This is a key optimization: the CPU prefetch core maps the buffer,
  // which loads the data into the shared LLC where GPU can access it
  cl_int err;
  void *mappedPtr = prefetchQueue.enqueueMapBuffer(
      *m_targetBuffer, CL_FALSE, // Non-blocking
      CL_MAP_WRITE, m_currentOffset, buf.size, nullptr, &buf.mapEvent, &err);

  if (err != CL_SUCCESS) {
    std::cerr << "[Prefetch] Map failed: " << err << "\n";
    buf.inUse = false;
    return;
  }

  // Set up callback to copy data and unmap
  buf.mapEvent.setCallback(CL_COMPLETE, [](cl_event, cl_int, void *userData) {
    // This callback runs when map completes
    // Data is now in cache
  }, &buf);

  // Wait for map to complete (this brings data to cache)
  buf.mapEvent.wait();

  // Copy data to the mapped region
  std::memcpy(mappedPtr, buf.hostPtr, buf.size);

  // Unmap to make data available to GPU
  prefetchQueue.enqueueUnmapMemObject(*m_targetBuffer, mappedPtr, nullptr,
                                      &buf.unmapEvent);

  m_currentOffset += buf.size;
  m_prefetchedChunks++;
  buf.ready = true;
}

int PrefetchManager::getReadyChunk(size_t &offset, size_t &size) {
  std::unique_lock<std::mutex> lock(m_mutex);

  for (size_t i = 0; i < m_buffers.size(); i++) {
    if (m_buffers[i].ready && m_buffers[i].inUse) {
      offset = m_buffers[i].offset;
      size = m_buffers[i].size;
      return static_cast<int>(i);
    }
  }

  return -1;
}

void PrefetchManager::markChunkProcessed(int chunkIndex) {
  std::unique_lock<std::mutex> lock(m_mutex);

  if (chunkIndex >= 0 && chunkIndex < static_cast<int>(m_buffers.size())) {
    m_buffers[chunkIndex].ready = false;
    m_buffers[chunkIndex].inUse = false;
    m_processedChunks++;

    // Prefetch next chunk if available
    if (m_currentOffset < m_totalSize) {
      lock.unlock();
      prefetchWorker();
    } else if (m_processedChunks >= m_totalChunks) {
      m_complete = true;
      m_cv.notify_all();
    }
  }
}

void PrefetchManager::waitAll() {
  std::unique_lock<std::mutex> lock(m_mutex);
  m_cv.wait(lock, [this] { return m_complete.load(); });

  // Wait for any pending unmap operations
  for (auto &buf : m_buffers) {
    if (buf.inUse) {
      try {
        buf.unmapEvent.wait();
      } catch (...) {
      }
    }
  }
}

bool PrefetchManager::isComplete() const { return m_complete; }

// ============================================================================
// Prefetch API Implementation
// ============================================================================

void prefetchInit() {
  // Use PE config for chunk size
  gPrefetchManager.init(peConfig.chunkSize, peConfig.numChunks);
}

void prefetchCleanup() { gPrefetchManager.cleanup(); }

void prefetchToDevice(cl::Buffer &deviceBuf, const void *hostData, size_t size,
                      cl::Event *completeEvent) {
  if (!peConfig.enablePrefetch) {
    // Fall back to synchronous copy
    prefetchSync(deviceBuf, hostData, size);
    return;
  }

  // Start chunked prefetch
  size_t numChunks =
      gPrefetchManager.startPrefetch(deviceBuf, hostData, size, 1);

  // Process all chunks
  for (size_t i = 0; i < numChunks; i++) {
    size_t offset, chunkSize;
    int chunkIdx;

    // Wait for a chunk to be ready
    while ((chunkIdx = gPrefetchManager.getReadyChunk(offset, chunkSize)) < 0) {
      std::this_thread::yield();
    }

    // Mark chunk as processed (this triggers next prefetch)
    gPrefetchManager.markChunkProcessed(chunkIdx);
  }

  gPrefetchManager.waitAll();
}

void prefetchWithCallback(cl::Buffer &deviceBuf, const void *hostData,
                          size_t size, PrefetchCallback callback,
                          void *userData) {
  if (!peConfig.enablePrefetch) {
    prefetchSync(deviceBuf, hostData, size);
    if (callback) {
      callback(0, size, 0, userData);
    }
    return;
  }

  size_t numChunks =
      gPrefetchManager.startPrefetch(deviceBuf, hostData, size, 1);

  for (size_t i = 0; i < numChunks; i++) {
    size_t offset, chunkSize;
    int chunkIdx;

    while ((chunkIdx = gPrefetchManager.getReadyChunk(offset, chunkSize)) < 0) {
      std::this_thread::yield();
    }

    // Invoke callback for this chunk
    if (callback) {
      callback(offset, chunkSize, chunkIdx, userData);
    }

    gPrefetchManager.markChunkProcessed(chunkIdx);
  }

  gPrefetchManager.waitAll();
}

void prefetchSync(cl::Buffer &deviceBuf, const void *hostData, size_t size) {
  // Simple synchronous write
  execGpuQueue.enqueueWriteBuffer(deviceBuf, CL_TRUE, 0, size, hostData);
}

