#pragma once

#define MAX_CHARS_FOR_SEQ1 3000
#define MAX_CHARS_FOR_SEQ2 2000
#define SIZE_GROUP 9
#define SIZE_SEMI_GROUP 11
#define SCORE_TYPE {MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT}
#define SCORE_BLOCK_LENGTH {1, 1, 1, 1}
#define SCORE_NUM_ATTRIBUTES 4
#define WEIGHT_TYPE {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE}
#define WEIGHT_BLOCK_LENGTH {1, 1, 1, 1}
#define WEIGHT_NUM_ATTRIBUTES 4



typedef struct {
	double w1;
	double w2;
	double w3;
	double w4;
} Weights;


typedef struct {
	int countStars;
	int countColons;
	int countDots;
	int countSpaces;
	double score;
} Counts;


typedef struct {
	double score;
	int mute_loc;
	int offset;
	int num_of_Seq;
} Scores;


int computeOnGPU(char *seq1, char *seq2, Weights* weights, Scores *scores);
void addMutantsEachSeq(char *seq2, int size_seq2, char **new_seq2, int *i, Scores **scores);
void readFromFile(const char *file_path, Weights *weights, char **seq1, int *numOfSeq2, char ***seq2);
void writeToFile(FILE* file, Scores* scores, int numOfSeq2);
int getSize1(char *s);
double compareSameLenSequences(Weights weights, char *seq1, char *seq2, int size_seq2);
void compareSequence(char *seq1, char *seq2, Counts *counts, Weights weights, Scores *scores);
void slave(char *one_seq, char *seq1);
double compareDiffrentLenSequences(Weights weights, char *seq1, char *seq2, int size_seq2, Scores **scores);
void substring(char s[], char *sub, int p, int l);
int calcMaxOffsetForEachSeq(char *seq1, char *seq2);
int checkSemiOrSemiConservativeGroups1(char *c1, char *c2, const char *group[], int size);
void Master(char **seq1, int *numOfSeq2, char ***seq2);

