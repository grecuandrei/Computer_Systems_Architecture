#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"
#define blockSize	1024
#define load_factor 0.8
using namespace std;


/**
 * Functie de hash folosita pentru implementarea programului
 */
__device__ int my_hash(int data, int limit) {
	return ((long)abs(data) * 41812097llu) % 16991387857llu % limit;
}

/*
 *	Se verifica daca cheia este 0 / diferita de 0, pentru fiecare pereche din noul hashmap
 *	daca este 0 inserez pe pozitia data de functia de hash, altfel trec 
 *	la urmatoarea pozitie
 */

__global__ void kernel_reshape(entry *hashmap, entry *newHashmap, int hashSize, int newSize)
{
	uint32_t hash, ok = 0;

	if (blockIdx.x * blockDim.x + threadIdx.x >= hashSize ||
	 hashmap[blockIdx.x * blockDim.x + threadIdx.x].key == 0)
		return;

	hash = my_hash(hashmap[blockIdx.x * blockDim.x + threadIdx.x].key, newSize);
	
	for(;ok != 1; hash += 1) {
		hash = hash % newSize;
		
		if (atomicCAS(&newHashmap[hash].key, 0, hashmap[blockIdx.x * blockDim.x + threadIdx.x].key) == 0 
		|| atomicCAS(&newHashmap[hash].key, 0, hashmap[blockIdx.x * blockDim.x + threadIdx.x].key) ==
		 hashmap[blockIdx.x * blockDim.x + threadIdx.x].key) {
			ok = 1;
			newHashmap[hash].value = hashmap[blockIdx.x * blockDim.x + threadIdx.x].value;
			return;
		}
	}
	return;
}

/*
 *	Se va adauga in hashmap asocierea cheie, valoare
 *	primita ca parametru
 */

__device__ void kernel_insert(entry *hashmap, uint32_t key, uint32_t value, uint32_t hashSize)
{
	uint32_t hash;
	hash = my_hash(key, hashSize);

	for(;;){
		hash = hash % hashSize;
		if (atomicCAS(&hashmap[hash].key, 0, key) == 0 || atomicCAS(&hashmap[hash].key, 0, key) == key) {
			hashmap[hash].value = value;
			return;
		}
		hash += 1;
	}
}

/*
 * 	Fiecare thread va lua (cheia, valoarea) ceruta de pozitia din
 *	vectorul de chei si valori primit ca parametru
 */

__global__ void gpu_insert_kernel(entry *hashmap, uint32_t *keys, uint32_t *values, uint32_t numKeys, uint32_t hashSize)
{
	if (numKeys > blockIdx.x * blockDim.x + threadIdx.x) {
		kernel_insert(hashmap, keys[blockIdx.x * blockDim.x + threadIdx.x],
		 values[blockIdx.x * blockDim.x + threadIdx.x], hashSize);
	}
	return;
}

/*
 *	Se va cauta cheia in hashmap, daca se gaseste pe pozitia indicata
 *	de functia de hash, se va introduce in vectorul de valori
 */

__global__ void kernel_get(entry *hashmap, uint32_t *keys, uint32_t *values, uint32_t numKeys, uint32_t hashSize)
{
	uint32_t hash;
	uint32_t hashKey;
	
	hashKey = keys[blockIdx.x * blockDim.x + threadIdx.x];
	hash = my_hash(hashKey, hashSize);

	if (blockIdx.x * blockDim.x + threadIdx.x >= numKeys 
	|| hashmap[hash].key == 0)
		return;
	
	for (;;hash += 1) {
		hash = hash % hashSize;
		if (hashmap[hash].key == hashKey) {
                        values[blockIdx.x * blockDim.x + threadIdx.x] = hashmap[hash].value;
                        return;
                }

		if (hashmap[hash].key == 0) {
			values[blockIdx.x * blockDim.x + threadIdx.x] = 0;
			return;
		}		
	}
}

GpuHashTable::GpuHashTable(int size) {
	uint32_t numBlocks =  size / blockSize;
	hashSize = size;
	pairs = 0;
	hashmap = nullptr;
	if (size % blockSize != 0)
		numBlocks++;

	glbGpuAllocator->_cudaMalloc((void **)&hashmap, size * sizeof(entry));
	cudaMemset(hashmap, 0, size * sizeof(entry));
	
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree(hashmap);
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	entry *newHashmap;
	uint32_t numBlocks;
	
	glbGpuAllocator->_cudaMalloc((void **)&newHashmap, numBucketsReshape * sizeof(entry));
	cudaMemset(newHashmap, 0, numBucketsReshape * sizeof(entry));
	
	if (!(hashSize % blockSize == 0))
		numBlocks = hashSize /  blockSize + 1;
	else 
		numBlocks = hashSize /  blockSize;

	
	kernel_reshape <<< numBlocks, blockSize >>> (hashmap, newHashmap, hashSize, numBucketsReshape);

	cudaDeviceSynchronize();
	glbGpuAllocator->_cudaFree(hashmap);
	hashmap = newHashmap;
	hashSize = numBucketsReshape;
}

float GpuHashTable::loadFactor() {
	return (float) pairs / hashSize;
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	uint32_t *dKeys, *dValues;
	uint32_t numBlocks;

	glbGpuAllocator->_cudaMalloc((void **)&dKeys, numKeys * sizeof(uint32_t));
	glbGpuAllocator->_cudaMalloc((void **)&dValues, numKeys * sizeof(uint32_t));

	if (!(hashSize % blockSize == 0))
                numBlocks = hashSize /  blockSize + 1;
        else
                numBlocks = hashSize /  blockSize;
	cudaMemcpy(dKeys, keys, numKeys * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dValues, values, numKeys * sizeof(uint32_t), cudaMemcpyHostToDevice);


	if (load_factor <= (numKeys + pairs) / hashSize) 
		reshape((numKeys + pairs) / 0.8f);
	gpu_insert_kernel <<< numBlocks, blockSize >>> (hashmap, dKeys, dValues,
		numKeys, hashSize);
	pairs = pairs + numKeys;
	cudaDeviceSynchronize();

	glbGpuAllocator->_cudaFree(dKeys);
	glbGpuAllocator->_cudaFree(dValues);

	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	uint32_t *deviceKeys, *deviceValues, numBlocks;
	glbGpuAllocator->_cudaMalloc((void **)&deviceKeys,  numKeys * sizeof(uint32_t));
	glbGpuAllocator->_cudaMalloc((void **)&deviceValues,  numKeys * sizeof(uint32_t));
	cudaMemcpy(deviceKeys, keys,  numKeys * sizeof(uint32_t), cudaMemcpyHostToDevice);
	int *values = (int *)malloc(numKeys * sizeof(int));

	if (!(numKeys % blockSize == 0))
		numBlocks = numKeys / blockSize + 1;
	else 
		numBlocks = numKeys / blockSize;
	
	kernel_get <<<numBlocks, blockSize >>> (hashmap, deviceKeys, deviceValues, numKeys, hashSize);
	cudaDeviceSynchronize();
	glbGpuAllocator->_cudaFree(deviceKeys);
	 cudaMemcpy(values, deviceValues,  numKeys * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	glbGpuAllocator->_cudaFree(deviceValues);
	
	return values;
}

