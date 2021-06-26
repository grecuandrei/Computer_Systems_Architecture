/*
 * Tema 2 ASC
 * 2021 Spring
 */
#include "utils.h"

/*
 * Add your optimized implementation here
 */
double* my_solver(int N, double *A, double* B) {
	double *C;
	double *B_t;
	double *A_t;
	double *AB;

	register int size = N * N * sizeof(*C);

	C = malloc(size);
	if (NULL == C)
		exit(EXIT_FAILURE);

	B_t = malloc(size);
	if (NULL == B_t)
		exit(EXIT_FAILURE);
	
	A_t = malloc(size);
	if (NULL == A_t)
		exit(EXIT_FAILURE);

	AB = malloc(size);
	if (NULL == AB)
		exit(EXIT_FAILURE);

	/* Transpunerea matricei B: B_t = B^t si A: A_t = A^t*/
	register int i;
	for (i = 0; i != N; ++i) {
		register double *B_t_ptr = B_t + i;  /* coloana i din B' */
		register double *A_t_ptr = A_t + i;  /* coloana i din A' */

		register double *B_ptr = B + i * N;  /* linia i din B */
		register double *A_ptr = A + i * N;  /* linia i din A */
		
		register int j;
		for (j = 0; j != N; ++j, A_t_ptr += N, ++A_ptr, B_t_ptr += N, ++B_ptr) {
			*B_t_ptr = *B_ptr;
			*A_t_ptr = *A_ptr;
		}
	}

	/* C = A×B×Bt + At×A */
	/* AB = A*B */
	register double *A_ptr = A;
	register double *B_ptr = B;
	register double *B_copy = B;  /* linia k din B */
	register double *A_copy = A;  /* coloana k din A */
	register double *AB_ptr;
	register int k;
	for (k = 0; k != N; ++k, B_copy += N, ++A_copy)
	{
		A_ptr = A_copy;
		register double *AB_copy = AB;
		register int i;

		for (i = 0; i != N; ++i, A_ptr += N, AB_copy += N)
		{
			AB_ptr = AB_copy;
			B_ptr = B_copy;

			register int j;

			for (j = 0; j != N; ++j, ++B_ptr, ++AB_ptr)
			{
				*AB_ptr += *A_ptr * *B_ptr;
			}
		}
	}

	/* C = AB * B_t */
	register double *C_ptr;
	register double *AB_copy = AB;  /* coloana k din AB */
	AB_ptr = AB;
	B_ptr = B_t;
	B_copy = B_t;  /* linia k din B */
	for (k = 0; k != N; ++k, B_copy += N, ++AB_copy)
	{
		AB_ptr = AB_copy;
		register double *C_copy = C;

		register int i;

		for (i = 0; i != N; ++i, AB_ptr += N, C_copy += N)
		{
			C_ptr = C_copy;  /* linia i din C */
			B_ptr = B_copy;

			register int j;

			for (j = 0; j != N; ++j, ++B_ptr, ++C_ptr)
			{
				*C_ptr += *AB_ptr * *B_ptr;
			}
		}
	}

	/* C = ABB_t + A_tA */
	for (i = 0; i != N; ++i) {
		register double *C_ptr = C + i * N;
		register double *A_tA_copy = A_t + i * N;
		register int j;

		for (j = 0; j != N; ++j, ++C_ptr) {
			/* rezultatul calulului pe o linie */
			register double result = 0;

			register double *A_tA_ptr = A_tA_copy;

			/* linia j din A */ 
			register double *A_ptr = A_t + j * N;

			register int k;

			for (k = 0; k != N; ++k, ++A_tA_ptr, ++A_ptr) {
				result += *A_tA_ptr * *A_ptr;
			}

			*C_ptr += result;
		}
	}

	free(B_t);
	free(A_t);
	free(AB);
	
	return C;	
}
