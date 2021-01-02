/*
 ============================================================================
 Name        : AnnaMeleshko.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello MPI World in C 
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include "myProto.h"
#include <omp.h>
#include <stddef.h>

void createScoreType(MPI_Datatype *scoreMpiType);
void createWeightType(MPI_Datatype *weightMpiType);

int getSize1(char *s) {
/********************
Get size of each seq.
*********************/
	char *t;
	int size = 0;
	for (t = s; *t != '\0'; t++) {
		size++;
	}

	return size;
}

void readFromFile(const char *file_path, Weights *weights, char **seq1,
		int *numOfSeq2, char ***seq2) {
/********************
Read data from file.
*********************/
	int i;
	int size_seq1;
	int size_seq2;
	FILE *file = fopen(file_path, "r");
	if (!file) {
	}
	i = fscanf(file, "%lf %lf %lf %lf", &weights->w1, &weights->w2,
			&weights->w3, &weights->w4);
	*seq1 = (char*) malloc(sizeof(char) * MAX_CHARS_FOR_SEQ1);
	i = fscanf(file, "%s", *seq1);
	size_seq1 = getSize1(*seq1);
	if(size_seq1 > MAX_CHARS_FOR_SEQ1){
		printf("\n****ERROR! cannot read the file, the string for Sequence 1 is too long of length:%d!****\n",size_seq1);
		exit(1);
	}
	
	i = fscanf(file, "%d", numOfSeq2);
	*seq2 = (char**) malloc(*numOfSeq2 * sizeof(char*));
	for (i = 0; i < *numOfSeq2; i++) {
		(*seq2)[i] = (char*) malloc(MAX_CHARS_FOR_SEQ2 * sizeof(char));
		fscanf(file, "%s", (*seq2)[i]);
		size_seq2 = getSize1((*seq2)[i]);
		if(size_seq2 > MAX_CHARS_FOR_SEQ2){
		printf("\n****ERROR! cannot read the file, the string for Sequence 2 is too long of length:%d!****\n",size_seq2);
		exit(1);
		}
	}
	fclose(file);
}


void writeToFile(FILE* file, Scores* scores, int numOfSeq2){
/********************
Write to file.
*********************/
	int i;
	for(i = 0 ; i < numOfSeq2 ; i++){
		fprintf(file,"Sequence Number: %d\t|Best offset(n): %d\t|Best MS(k): %d\t\n",scores[i].num_of_Seq, scores[i].mute_loc, scores[i].offset);
	}
}


int checkSemiOrSemiConservativeGroups1(char *c1, char *c2, const char *group[],
		int size) {
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
		size_g = getSize1((char*) group[i]);
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



int calcMaxOffsetForEachSeq(char *seq1, char *seq2) {
/****************************************************
Method that calculates max offset for each sequence.
*****************************************************/
	int size_seq1;
	int size_seq2;
	int maxOffsetEachSeq2;
	size_seq1 = getSize1(seq1);
	size_seq2 = getSize1(seq2);
	maxOffsetEachSeq2 = abs(size_seq1 - size_seq2);
	return maxOffsetEachSeq2;
}


void addMutantsEachSeq(char *seq2, int size_seq2, char **new_seq2, int *i,
		Scores **scores) {
/**************************************
Method that adds mutants to sequence.
***************************************/
	int j, k;
	int size_new_seq = size_seq2 + 1;
	*new_seq2 = (char*)calloc(size_new_seq, sizeof(char));
	for (j = 0, k = 0; j < size_seq2 + 1; j++, k++) {
		if (*i == j) {
			(*new_seq2)[*i] = '-';
			(*scores)->mute_loc = *i;
			j += 1;
		}
		(*new_seq2)[j] = seq2[k];
	}
}


void compareSequence(char *seq1, char *seq2, Counts *counts, Weights weights,
		Scores *scores) {
/*********************************************************************************************
Method that compares sequences and stores the final score, and all the details to the struct.
**********************************************************************************************/

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int j;
	int size_seq1;
	int size_seq2;
	char *new_seq2;
	int size_new_seq;
	int mute_loc = 0;
	int offset = 0;
	scores->score = 0.0;
	scores->mute_loc = 0;
	scores->offset = 0;
	size_seq1 = getSize1(seq1);
	double maxScore = -INFINITY;
	size_seq2 = getSize1(seq2);
	size_new_seq = size_seq2 + 1;
	if (size_seq1 == size_seq2) {// check if the sequences ar the same size
		for (j = 0; j < size_new_seq; j++) {
				addMutantsEachSeq(seq2, size_seq2, &new_seq2, &j, &scores);
				scores->score = compareSameLenSequences(weights, seq1,
						new_seq2, size_seq2);
			if (scores->score > maxScore) {
				maxScore = scores->score;
			}
		}
	} else {
		for (j = 0; j < size_new_seq; j++) {// if not
			addMutantsEachSeq(seq2, size_seq2, &new_seq2, &j, &scores);// than add mutant to each sequence
			scores->score = compareDiffrentLenSequences( weights,
					seq1, new_seq2, size_seq2, &scores);// and then compare sequences with diffrent length 
			if (scores->score > maxScore) {//store the max score, offset and mutant location
				maxScore = scores->score;
				mute_loc = scores->mute_loc;
				offset = scores->offset;
			}
		}// store the values to the Score struct
		scores->score = maxScore;
		scores->mute_loc = mute_loc;
		scores->offset = offset;
	}
}


double compareDiffrentLenSequences(Weights weights, char *seq1,
		char *seq2, int size_seq2, Scores **scores) {
/**********************************************************************************************************************
Method that compares diffrent length sequences - compares every possible offset of the sequences.
this method using OPEN MP, where each thread is responisble of other sequence offset, i set the number of threads to 4.
***********************************************************************************************************************/
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int maxOffset;
	int i;
	int tid = 0;
	char *sub;
	double max_score = -INFINITY;

	double *max_scores_each_t;
	int *offset_each_t;
	int n_ot = 4;
	omp_set_num_threads(n_ot);
	max_scores_each_t = (double*)calloc(n_ot, sizeof(double));
	offset_each_t =(int*)calloc(n_ot, sizeof(int));
	(*scores)->offset = 0;
	maxOffset = calcMaxOffsetForEachSeq(seq1, seq2);
#pragma omp parallel private(i, tid, sub)
	{
		sub = (char*)malloc(size_seq2 * sizeof(char));
		tid = omp_get_thread_num();
		max_scores_each_t[tid] = -INFINITY;
		#pragma omp for
		for (i = 0; i < maxOffset + 1; i++) {
			double score = 0.0;
			substring(seq1, sub, i + 1, size_seq2);
			score = compareSameLenSequences(weights, sub, seq2,
					size_seq2);
			if (score > max_scores_each_t[tid]) {
				max_scores_each_t[tid] = score;
				offset_each_t[tid] = i;
			}
		}
	}
	for (i = 0; i < n_ot; i++) {
		if (max_scores_each_t[i] > max_score) {
			max_score = max_scores_each_t[i];
			(*scores)->offset = offset_each_t[i];
		}
	}
	return max_score;
}


double compareSameLenSequences(Weights weights, char *seq1,
		char *seq2, int size_seq2) {
/*******************************************************************************
Method that compares sequences of same length and count signs for each sequence 
and than calculates the score with weights formule.
********************************************************************************/
	const char *cons_Groups[9] = { "NDEQ", "MILV", "FYM", "NEQK", "QHRK", "HY",
			"STA", "NHQK", "MILF" };
	const char *semi_Cons_Groups[11] = { "SAG", "SGND", "NEQHRK", "ATV", "STPA",
			"NDEQHK", "HFY", "CSA", "STNK", "SNDEQK", "FVLIM" };
	int i;
	Counts counts;
	double score = 0.0;
	counts.countStars = 0;
	counts.countDots = 0;
	counts.countColons = 0;
	counts.countSpaces = 0;
	for (i = 0; i < size_seq2; i++) {
		if (seq1[i] == seq2[i]) {
			counts.countStars++;
		} else if (checkSemiOrSemiConservativeGroups1(&(seq1)[i], &(seq2)[i],
				cons_Groups, 9) == 1) {
			counts.countColons++;
		} else if (checkSemiOrSemiConservativeGroups1(&seq1[i], &(seq2)[i],
				semi_Cons_Groups, 11) == 1) {
			counts.countDots++;
		} else {
			counts.countSpaces++;
		}
	}
	score = (weights.w1 * (double) counts.countStars)
			- (weights.w2 * (double) counts.countColons)
			- (weights.w3 * (double) counts.countDots)
			- (weights.w4 * (double) counts.countSpaces);
	return score;
}

void substring(char *s, char *sub, int p, int l) {
/*****************************************
Method that gives a substring of a string.
******************************************/
	int c = 0;

	while (c < l) {
		sub[c] = s[p + c - 1];
		c++;
	}
	sub[c] = '\0';
}



void createWeightType(MPI_Datatype *weightMpiType) {
/*****************************************
Method that creates MPI Weight DataType
******************************************/
	int blocklengths[WEIGHT_NUM_ATTRIBUTES] = WEIGHT_BLOCK_LENGTH;
	MPI_Datatype types[WEIGHT_NUM_ATTRIBUTES] = WEIGHT_TYPE;
	MPI_Aint offsets[WEIGHT_NUM_ATTRIBUTES];
	offsets[0] = offsetof(Weights, w1);
	offsets[1] = offsetof(Weights, w2);
	offsets[2] = offsetof(Weights, w3);
	offsets[3] = offsetof(Weights, w4);
	MPI_Type_create_struct(WEIGHT_NUM_ATTRIBUTES, blocklengths, offsets, types,
			weightMpiType);
	MPI_Type_commit(weightMpiType);
}


void Master(char **seq1, int *numOfSeq2, char ***seq2) {
/***********************************************************************************
Method for the Master(rank = 0), that reads from file, than gives job to slaves - 
gives every slave a sequence to work on, and than waits for the results.
************************************************************************************/
	int i;
	int size;
	MPI_Datatype weightMPIType;
	createWeightType(&weightMPIType);
	int size_seq2;
	int size_seq1;
	MPI_Status status;
	MPI_Datatype scoreMPIType;
	createScoreType(&scoreMPIType);
	Scores scores;
	Weights weights;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	readFromFile("input.txt", &weights, seq1, numOfSeq2, seq2);
	size_seq1 = getSize1(*seq1);
	int task_count = 0;
	int term_count = 0;
	int TERMINATION_TAG = 1;
	MPI_Bcast(&size_seq1, 1, MPI_INT, 0, MPI_COMM_WORLD);//give slaves all the information needed:num of sequences, weights, seq1 etc.
	MPI_Bcast(*seq1, size_seq1, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Bcast(&weights, 1, weightMPIType, 0, MPI_COMM_WORLD);
	MPI_Bcast(numOfSeq2, 1, MPI_INT, 0, MPI_COMM_WORLD);
	const char* file_path;
	file_path = "output.txt";
	FILE *file = fopen(file_path, "w");
	int num_seq;
	Scores* scores_arr;
	scores_arr = (Scores*)malloc(*numOfSeq2*sizeof(Scores));
	if (size - 1 == *numOfSeq2) {//check if num of processes equal to num of sequences
		for (i = 1; i < size; i++) {
			size_seq2 = getSize1((*seq2)[i - 1]);
			MPI_Send(&size_seq2, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send((*seq2)[i - 1], size_seq2, MPI_CHAR, i, 0, MPI_COMM_WORLD);
		}

		for (i = 1; i < size; i++) {
			MPI_Recv(&num_seq, 1, MPI_INT, MPI_ANY_SOURCE, 0,
			MPI_COMM_WORLD, &status);
			MPI_Recv(&scores, 1, scoreMPIType, MPI_ANY_SOURCE, 0,
			MPI_COMM_WORLD, &status);
			scores.num_of_Seq = num_seq;
			scores_arr[num_seq -1] = scores;
			MPI_Send(&term_count, 1, MPI_INT, status.MPI_SOURCE, TERMINATION_TAG,
			MPI_COMM_WORLD);
		}
	} else if (*numOfSeq2 > size - 1) {//check if num of processes less than num of sequences
		for (i = 1; i < size; i++) {
			size_seq2 = getSize1((*seq2)[i - 1]);
			MPI_Send(&size_seq2, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			task_count++;
			MPI_Send(&task_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send((*seq2)[i - 1], size_seq2, MPI_CHAR, i, 0, MPI_COMM_WORLD);
			
		}
		do {// if have less processes than sequences -than same slaves need to do more job,so the master will wait for the result,
		    //and check if have more tasks to send, if have will send it to the first slave that returned the reslut, if dont have will send termonation tag.
			MPI_Recv(&num_seq, 1, MPI_INT, MPI_ANY_SOURCE, 0,
			MPI_COMM_WORLD, &status);
			MPI_Recv(&scores, 1, scoreMPIType, MPI_ANY_SOURCE, 0,
			MPI_COMM_WORLD, &status);
			scores.num_of_Seq = num_seq;
			scores_arr[num_seq -1] = scores;
			if (task_count < *numOfSeq2) {
				size_seq2 = getSize1((*seq2)[task_count]);
				task_count++;
				MPI_Send(&size_seq2, 1, MPI_INT, status.MPI_SOURCE, 0,MPI_COMM_WORLD);
				MPI_Send(&task_count, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
				MPI_Send((*seq2)[task_count-1], size_seq2, MPI_CHAR,
						status.MPI_SOURCE, 0, MPI_COMM_WORLD);
				
			} else {
				MPI_Send(&term_count, 1, MPI_INT, status.MPI_SOURCE, TERMINATION_TAG,
				MPI_COMM_WORLD);
				term_count++;
			}
		} while (term_count < size - 1);
	} else {//check if num of processes grather than num of sequences
		for (i = 1; i < size; i++) {
			if (i > *numOfSeq2) {
				MPI_Send(&term_count, 1, MPI_INT, i, TERMINATION_TAG,
				MPI_COMM_WORLD);
			} else {
				size_seq2 = getSize1((*seq2)[i - 1]);
				MPI_Send(&size_seq2, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send((*seq2)[i - 1], size_seq2, MPI_CHAR, i, 0,
						MPI_COMM_WORLD);
			}
		}
		for (i = 1; i < *numOfSeq2 + 1; i++) {
			MPI_Recv(&num_seq, 1, MPI_INT, MPI_ANY_SOURCE, 0,MPI_COMM_WORLD, &status);
			MPI_Recv(&scores, 1, scoreMPIType, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			scores.num_of_Seq = num_seq;
			scores_arr[num_seq -1] = scores;
			MPI_Send(&term_count, 1, MPI_INT, status.MPI_SOURCE, TERMINATION_TAG,
							MPI_COMM_WORLD);
		}
	}
	writeToFile(file, scores_arr, *numOfSeq2);
	fclose(file);
}


void createScoreType(MPI_Datatype *scoreMpiType) {
/*****************************************
Method that creates MPI Weight DataType
******************************************/
	int blocklengths[SCORE_NUM_ATTRIBUTES] = SCORE_BLOCK_LENGTH;
	MPI_Datatype types[SCORE_NUM_ATTRIBUTES] = SCORE_TYPE;
	MPI_Aint offsets[SCORE_NUM_ATTRIBUTES];
	offsets[0] = offsetof(Scores, score);
	offsets[1] = offsetof(Scores, mute_loc);
	offsets[2] = offsetof(Scores, offset);
	offsets[3] = offsetof(Scores, num_of_Seq);
	MPI_Type_create_struct(SCORE_NUM_ATTRIBUTES, blocklengths, offsets, types, scoreMpiType);
	MPI_Type_commit(scoreMpiType);
}


void slave(char *one_seq, char *seq1) {
/***********************************************************************************
Method for the Slaves(rank != 0), each slave gets a job from the master -
every slave get a sequence to work on, and than sends the results back to the master.
************************************************************************************/
	int dest;
	dest = 0;
	int my_rank;
	int size_seq1;
	int numOfSeq2 = 0;
	Scores scores;
	Weights weights;
	Counts counts;
	MPI_Datatype scoreMPIType;
	createScoreType(&scoreMPIType);
	MPI_Datatype weightMPIType;
	createWeightType(&weightMPIType);
	scores.mute_loc = 0;
	scores.score = 0.0;
	scores.offset = 0;
	int termination_tag = 1;
	MPI_Status status; /* return status for receive */
	int tag = 0; /* tag for messages */
	int size;
	int num_seq;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Bcast(&size_seq1, 1, MPI_INT, 0, MPI_COMM_WORLD);
	seq1 = (char*)calloc(size_seq1, sizeof(char));
	MPI_Bcast(seq1, size_seq1, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Bcast(&weights, 1, weightMPIType, 0, MPI_COMM_WORLD);
	MPI_Bcast(&numOfSeq2, 1, MPI_INT, 0, MPI_COMM_WORLD);
	do {//slaves receives job until the master sends a termination tag which means he dont have more job to send
		MPI_Recv(&size, 1, MPI_INT, dest, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG == termination_tag) {
			return;
		}
		MPI_Recv(&num_seq, 1, MPI_INT, dest, MPI_ANY_TAG, MPI_COMM_WORLD, &status);// recieve the number of the current sequence
		one_seq = (char*)calloc(size, sizeof(char));
		MPI_Recv(one_seq, size, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &status);
		if(my_rank % 2 == 0){
		compareSequence(seq1, one_seq, &counts, weights, &scores);
		}else{
		computeOnGPU(seq1, one_seq, &weights, &scores);
		}
		MPI_Send(&num_seq, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&scores, 1, scoreMPIType, 0, 0, MPI_COMM_WORLD);
	} while (1);
}


int main(int argc, char *argv[]) {

	int my_rank; /* rank of process */
	int p; /* number of processes */
	char *seq1;
	int numOfSeq2;
	char **seq2;
	char *one_seq = NULL;
	double t1, t2, time;

	/* start up MPI */

	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	t1 = MPI_Wtime();

	if (my_rank != 0) { //Slaves

		slave(one_seq, seq1);

	} else { //Master

		Master(&seq1, &numOfSeq2, &seq2);
		t2 = MPI_Wtime();
		time = (t2 - t1) / 60; // get the execution time in minutes
		printf("Executation time = %1.3f m \n", time);
	}

	MPI_Finalize();

	return 0;
}
