/*
 * Tema 2 ASC
 * 2021 Spring
 */
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include "cblas.h"

/* 
 * Add your BLAS implementation here
 */
double* my_solver(int N, double *A, double *B) {
	double *C;
	double *AA;

	C = malloc(N * N * sizeof(*C));
	if (NULL == C)
		exit(EXIT_FAILURE);

	AA = malloc(N * N * sizeof(*AA));
	if (NULL == AA)
		exit(EXIT_FAILURE);

	/* C= A×B×Bt + At×A */
	/* C = A*B */
	memcpy(C, B, N * N * sizeof(*C));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		C, N
	);

	/* AA = A_t*A */
	memcpy(AA, A, N * N * sizeof(*C));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		AA, N
	);

	/* C = C*B_t + A_tA */
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0, C, N,
		B, N,
		1.0, AA, N
	);

	free(C);

	return AA;
}
