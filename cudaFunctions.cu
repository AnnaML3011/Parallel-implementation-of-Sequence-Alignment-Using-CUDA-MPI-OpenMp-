#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"


__global__ void compareSequence2(char *seq1, char *seq2, Weights* weights, Scores *scores, double *results);
__device__ int getSize(char *s);
__device__ int checkSemiOrSemiConservativeGroups(char *c1, char *c2, const char *group[], int size);
__host__ cudaError_t checkForErrors(const char method_name[]);


__device__ int getSize(char *s) 
{
/*************************
Get the size of each seq. 
**************************/
	//get size of seq
	char *t;
	int size = 0;
	for (t = s; *t != '\0'; t++) {
		size++;
	}

	return size;
}


__host__ cudaError_t checkForErrors(const char method_name[])
{
/*****************************************************************************
Method that checks if there is any errors occurs while computing on the gpu. 
******************************************************************************/
	cudaError_t cudaStatus;
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%s launch failed: %s\n", method_name, cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateResultsKernel!\n", cudaStatus);
		return cudaStatus;
	}
	return cudaStatus;
}


int computeOnGPU(char *seq1, char *seq2, Weights* weights, Scores *scores) 
{
/**********************************************************************************************
Method that allocates memory on gpu for the needed parametrers, copy them to the gpu, 
and then call a function that making the comprasion between sequences, returning the results,
and than copy the paramaeters back to the cpu.
***********************************************************************************************/
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    size_t size_seq1 = getSize1(seq1);
    size_t size_seq2 = getSize1(seq2);
    char* c_seq1;
    char* c_seq2;
    double* results;
    Scores* c_scores;
    Weights* c_weights;
  
    // Allocate memory on GPU to copy the data from the host
    err = cudaMalloc((void **)&c_seq1, (size_seq1+1)*sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   err = cudaMalloc((void **)&c_seq2, (size_seq2+1)*sizeof(char));
     if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  	err = cudaMalloc((void **)&results, (size_seq2+1)*sizeof(double));
     if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
   }

    err = cudaMalloc((void **)&c_scores, sizeof(Scores));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   err = cudaMalloc((void **)&c_weights, sizeof(Weights));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy data from host to the GPU memory
    err = cudaMemcpy(c_seq1, seq1, (size_seq1+1)*sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(c_seq2, seq2, (size_seq2+1)*sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(c_scores, scores, sizeof(Scores), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(c_weights, weights, sizeof(Weights), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Kernel
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	int numThreads = props.maxThreadsPerBlock < size_seq2 ? props.maxThreadsPerBlock : size_seq2;
	int numBlocks = size_seq2 / numThreads;
	int extraBlock = size_seq2 % numThreads != 0;
	compareSequence2<<<numBlocks+extraBlock, numThreads>>>(c_seq1, c_seq2, c_weights, c_scores, results);

	err = checkForErrors("calculateResultsKernel");
	if (err != cudaSuccess) {
		return err;
	}

    // Copy the  result from GPU to the host memory.
 	err = cudaMemcpy(seq1, c_seq1, (size_seq1+1)*sizeof(char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result array from device to host - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(seq2, c_seq2, (size_seq2+1)*sizeof(char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result array from device to host - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(scores, c_scores, sizeof(Scores), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result array from device to host - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free allocated memory on GPU
    if (cudaFree(c_seq1) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	if (cudaFree(c_seq2) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }	
	if (cudaFree(results) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }		

	if (cudaFree(c_scores) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	if (cudaFree(c_weights) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return 0;
}


__device__ int checkSemiOrSemiConservativeGroups(char *c1, char *c2, const char *group[],int size) 
{
/**********************************************************************************************
Method that checks if 2 chars are in some conservative group, if so, returns 1, else returns 0.
***********************************************************************************************/
	int i, j;
	char *found_c1;
	char *found_c2;
	int size_g;
	for (i = 0; i < size; i++) {
		found_c1 = NULL;
		found_c2 = NULL;
		size_g = getSize((char*) group[i]);
		for (j = 0; j < size_g; j++) {
			if (found_c1 == NULL && *c1 == group[i][j]) {
				found_c1 = c1;
			}
			if (found_c2 == NULL && *c2 == group[i][j]) {
				found_c2 = c2;
			}
			if (found_c1 != NULL && found_c2 != NULL) {
				return 1;
			}
		}
	}
	return 0;
}


__global__ void compareSequence2(char *seq1, char *seq2, Weights* weights, Scores *scores, double *results) 
{
/**********************************************************************************************
Method that compares 2 strings with all the possible offsets and mutant locations.
number of threads is set to the length of current sequence 2, and each thread is checking each char 
of seq2 with all the possible mutants and offsets scores.
than only one thread sums the results and cheks if its the best score found.
***********************************************************************************************/
	int i,j,id,seq1_index,k,mute_loc,offset;
	int seq1_size = getSize(seq1);
	int seq2_size = getSize(seq2);
	double max_score =-INFINITY;
	double score = 0;
	const char *cons_Groups[9] = { "NDEQ", "MILV", "FYM", "NEQK", "QHRK", "HY",
			"STA", "NHQK", "MILF" };
	const char *semi_Cons_Groups[11] = { "SAG", "SGND", "NEQHRK", "ATV", "STPA",
			"NDEQHK", "HFY", "CSA", "STNK", "SNDEQK", "FVLIM" };
	id = (blockDim.x*blockIdx.x) + threadIdx.x;
	if(id > seq2_size )
		return;

	for(i=0;i<abs(seq1_size-seq2_size);i++)//for the possible offsets
	{
		for(j=0;j<seq2_size;j++)//for the possible mutants
		{
			seq1_index = id+i;
			if(id > j)
				seq1_index++;

			if (seq1[seq1_index] == seq2[id]) {
					results[id] = weights->w1;
			} else if (checkSemiOrSemiConservativeGroups(&(seq1)[seq1_index], &(seq2)[id],
					cons_Groups, 9) == 1) {
					results[id] = -weights->w2;
			} else if (checkSemiOrSemiConservativeGroups(&(seq1)[seq1_index], &(seq2)[id],
					semi_Cons_Groups, 11) == 1) {
					results[id] = -weights->w3;
			} else {
					results[id] = -weights->w4;
			}

			__syncthreads();//waits for all the threads to finish

			if(id == 0)// only one thread to sum the results
			{

				score = -weights->w4;// for the '-' that didnt include while checking
				for(k = 0; k < seq2_size ; k++)// sum all the results
				{
					score+= (double)results[k];
				}
				if(score > max_score)// check if the current score grather than max score 
				{
					max_score = score;
					mute_loc = j+1;
					offset = i;

				}
			}
			__syncthreads();
		}

	}

	if(id == 0)
	{
		scores->score = max_score;
		scores->mute_loc = mute_loc;
		scores->offset = offset;
	}
}


