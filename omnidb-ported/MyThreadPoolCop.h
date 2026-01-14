#ifndef THREAD_POOL_COP_H
#define THREAD_POOL_COP_H

//multiple threads for co-processing.

#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __linux__
#include <sched.h>
#endif

#define NUM_CORE_CPU 4
#define MAX_RUN ((1<<(NUM_CORE_CPU+1))-1)

struct threadPar{
	int threadid;
	void init(int pthreadid)
	{		
		threadid=pthreadid;
	}
};

// Thread function type for POSIX
typedef void* (*thread_func_t)(void*);

struct MyThreadPoolCop
{
public:
	void destory()
	{
		if(threadIds!=NULL)
			free(threadIds);
		if(masks!=NULL)
			free(masks);
		if(tParam!=NULL)
			free(tParam);
		if(pfnThreadProc!=NULL)
			free(pfnThreadProc);
	}
	// the number of threads
	int numThread;
	// the mask assigning threads to which core/processors
	int* masks;
	pthread_t *threadIds;
	void ** tParam;
	thread_func_t* pfnThreadProc;

//functions.
	int setCPUID(int tid, int cpuid)
	{
		int maskValue=(1<<cpuid)%MAX_RUN;
		masks[tid]=maskValue;
		return 0;
	}

	int assignTask(int tid, thread_func_t pfn)
	{
		pfnThreadProc[tid]=pfn;
		return 0;
	}

	int assignParameter(int tid, void* pvParam)
	{
		tParam[tid]=pvParam;
		return 0;
	}

	int run(void)
	{
		int i=0;
		for( i=0; i<numThread; i++ )
		{
			int result = pthread_create(&threadIds[i], NULL, pfnThreadProc[i], tParam[i]);
	 
			// Set CPU affinity if available (Linux-specific)
#ifdef __linux__
			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			// Set affinity based on mask
			for(int cpu = 0; cpu < NUM_CORE_CPU; cpu++) {
				if(masks[i] & (1 << cpu)) {
					CPU_SET(cpu, &cpuset);
				}
			}
			pthread_setaffinity_np(threadIds[i], sizeof(cpu_set_t), &cpuset);
#endif
	 
			if (result != 0) 
			{
				return i;
			}
		}
		// Wait until all threads have terminated.
		for(i=0; i<numThread; i++)
		{
			pthread_join(threadIds[i], NULL);
		}
		return 0;
	}

	void create(int nThead)
	{
		numThread=nThead;
		threadIds=(pthread_t*)malloc(sizeof(pthread_t)*numThread);
		masks=(int*)malloc(sizeof(int)*numThread);
		int i=0;
		for(i=0;i<numThread;i++)
			masks[i]=(1<<i)%MAX_RUN;
		tParam=(void**)malloc(sizeof(void*)*numThread);
		pfnThreadProc=(thread_func_t*)malloc(sizeof(thread_func_t)*numThread);
	}
};

#endif
