#ifndef _HASHCPU_
#define _HASHCPU_

#include <vector>

using namespace std;

#define cudaCheckError() { \
	cudaError_t e=cudaGetLastError(); \
	if(e!=cudaSuccess) { \
		cout << "Cuda failure " << __FILE__ << ", " << __LINE__ << ", " << cudaGetErrorString(e); \
		exit(0); \
	 }\
}
struct entry {
	uint32_t key;
	uint32_t value;
};


/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	public:
		int hashSize;
		entry *hashmap;
		int pairs;
		
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		float loadFactor();
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		~GpuHashTable();
};

#endif

