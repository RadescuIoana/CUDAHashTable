#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

__global__ void insert(GpuHashTable::hashCell *deviceHashTable,
						unsigned int *keys,
						unsigned int *values,
						unsigned int slotsElems,
						int numKeys) {

	int index, position, idx;
	unsigned int old;

	idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= numKeys)
		return;

	position = myHash(keys[idx], slotsElems);
	index = position;
	while (index < slotsElems) {
		old = atomicCAS(&deviceHashTable[index].key, (unsigned int) 0, keys[idx]);
		if (old == 0 || old == keys[idx]) {
			deviceHashTable[index].value = values[idx];
			return;
		}	
		index++;
	}
	index = 0;
	while (index < position) {
		old = atomicCAS(&deviceHashTable[index].key, (unsigned int) 0, keys[idx]);
		if (old == 0 || old == keys[idx]) {
			deviceHashTable[index].value = values[idx];
			return;
		}	
		index++;
	}
}

__global__ void reinsert(GpuHashTable::hashCell *newHashTable,
						GpuHashTable::hashCell *copyHashTable,
						unsigned int oldSize,
						unsigned int slotsElems) {
	int index, position, idx;
	unsigned int old;
	
	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= oldSize)
		return;
	
	if (copyHashTable[idx].key == 0)
		return;

	position = myHash(copyHashTable[idx].key, slotsElems);
	index = position;
	while (index < slotsElems) {
		old = atomicCAS(&newHashTable[index].key, (unsigned int) 0, copyHashTable[idx].key);
		if (!old || old == copyHashTable[idx].key) {
			newHashTable[index].value = copyHashTable[idx].value;
			return;
		}	
		index++;
	}
	index = 0;
	while (index < position) {
		old = atomicCAS(&newHashTable[index].key, (unsigned int) 0, copyHashTable[idx].key);
		if (!old || old == copyHashTable[idx].key) {
			newHashTable[index].value = copyHashTable[idx].value;
			return;
		}	
		index++;
	}
}

__global__ void get(GpuHashTable::hashCell *deviceHashTable,
					unsigned int *keys,
					unsigned int *values,
					unsigned int slotsElems,
					int numKeys) {
	int index, position, idx;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numKeys)
		return;

	position = myHash(keys[idx], slotsElems);
	index = position;
	while (index < slotsElems) {
		if (deviceHashTable[index].key == keys[idx]) {
			values[idx] = deviceHashTable[index].value;
			return;
		}	
		index++;
	}
	index = 0;
	while (index < position) {
		if (deviceHashTable[index].key == keys[idx]) {
			values[idx] = deviceHashTable[index].value;
			return;
		}	
		index++;
	}
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {

	cudaMalloc((void **) &hashTable, size * sizeof(hashCell));
	cudaMemset(hashTable, 0, size * sizeof(hashCell));

	slotsElems = size;
	numElems = 0;
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashTable);
	slotsElems = 0;
	numElems = 0;
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashCell *copyHashTable;

	if (numElems) {
		int mingridsize, threadblocksize, gridsize;
		cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, reinsert, 0, 0);
		
		cudaMalloc(&copyHashTable, slotsElems * sizeof(hashCell));
		cudaMemcpy(copyHashTable, hashTable, slotsElems * sizeof(hashCell), cudaMemcpyDeviceToDevice);

		cudaFree(hashTable);

		cudaMalloc((void **) &hashTable, numBucketsReshape * sizeof(hashCell));
		cudaMemset(hashTable, 0, numBucketsReshape * sizeof(hashCell));

		gridsize = ((unsigned int)slotsElems + threadblocksize - 1) / threadblocksize;
		reinsert<<<gridsize, threadblocksize>>> (hashTable, copyHashTable, slotsElems, numBucketsReshape);
		cudaDeviceSynchronize();
		slotsElems = numBucketsReshape;

		cudaFree(copyHashTable);
		return;
	}

	cudaFree(hashTable);

	cudaMalloc((void **) &hashTable, numBucketsReshape * sizeof(hashCell));
	cudaMemset(hashTable, 0, numBucketsReshape * sizeof(hashCell));
	slotsElems = numBucketsReshape;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int mingridsize;
	int threadblocksize;
	
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, insert, 0, 0);
	int gridsize = ((unsigned int)numKeys + threadblocksize - 1) / threadblocksize;

	unsigned int *deviceKeys, *deviceValues;

	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&deviceValues, numKeys * sizeof(int));
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	if ((float)(numElems + numKeys) / slotsElems > 0.95f)
		reshape((numElems + numKeys) * 1.25f);

	insert<<<gridsize, threadblocksize>>> (hashTable, deviceKeys, deviceValues, slotsElems, numKeys);
	cudaDeviceSynchronize();

	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	numElems += numKeys;
	return true;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	unsigned int *deviceKeys, *deviceValues;
	int *hostValues;

	int mingridsize, threadblocksize, gridsize;
	cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, get, 0, 0);

	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&deviceValues, numKeys * sizeof(int));
	cudaMemset(deviceValues, 0, numKeys * sizeof(int));

	hostValues = (int *) malloc(numKeys * sizeof(int));

	gridsize = ((unsigned int)numKeys + threadblocksize - 1) / threadblocksize;
	get<<<gridsize, threadblocksize>>> (hashTable, deviceKeys, deviceValues, slotsElems, numKeys);
	cudaDeviceSynchronize();
	cudaMemcpy(hostValues, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);	

	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	return hostValues;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	float loadFactor = 0.f;

	loadFactor = (float) numElems / slotsElems;

	return loadFactor; // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
