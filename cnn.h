#ifndef CNN_H
#define CNN_H

#include <stdlib.h>

typedef struct
{
  size_t rows;
  size_t cols;
  float *es;
} Matrix;

Matrix new_matrix(size_t rows, size_t cols);
void matrix_dot_product(Matrix dst, Matrix a, Matrix b);
void matrix_sum(Matrix dst, Matrix a);
void matrix_randomise(Matrix m, float lb, float ub);
void matrix_randomise_int(Matrix m, int lb, int ub);
void print_matrix(Matrix m, const char *name);

#define MATRIX_PRINT(m) print_matrix(m, #m)

float random_float(float lb, float ub);
int random_int(int lb, int ub);

#endif // CNN_H
