/*
 * Tema 2 ASC
 * 2021 Spring
 */
#include "utils.h"

/*
 * Add your unoptimized implementation here
 */
double* my_solver(int N, double *A, double* B) {
	double *C;
	double *ABB_t;
	double *A_tA;
	double *AB;
	int i, j, k;

	C = malloc(N * N * sizeof(*C));
	if (NULL == C)
		exit(EXIT_FAILURE);

	ABB_t = calloc(N * N, sizeof(*ABB_t));
	if (NULL == ABB_t)
		exit(EXIT_FAILURE);

	A_tA = calloc(N * N, sizeof(*A_tA));
	if (NULL == A_tA)
		exit(EXIT_FAILURE);

	AB = calloc(N * N, sizeof(*AB));
	if (NULL == AB)
		exit(EXIT_FAILURE);

	/* C = A×B×Bt + At×A */
	/* AB = A*B */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = i; k < N; k++)
				AB[i * N + j] += A[i * N + k] * B[k * N + j];

	/* ABB_t = AB * B_t */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];

	/* A_tA = A_t * A */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				A_tA[i * N + j] += A[k * N + i] * A[k * N + j];

	/* C = ABB_t + A_tA */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			C[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];

	free(ABB_t);
	free(A_tA);
	free(AB);

	return C;
}
