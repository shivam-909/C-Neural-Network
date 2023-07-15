#include "cnn.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_ELEM_AT(m, i, j) ((m).es[(i) * (m).cols + (j)])

Matrix new_matrix(size_t rows, size_t cols)
{
  Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.es = malloc(sizeof(*m.es) * rows * cols);
  return m;
}

float random_float(float lb, float ub)
{
  return ((float)rand() / (float)RAND_MAX) * (ub - lb) + lb;
}

int random_int(int lb, int ub) { return (int)random_float(lb, ub); }

void ordered_matrix_dot_product(Matrix dst, Matrix a, Matrix b)
{
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);

  for (size_t i = 0; i < a.rows * b.cols; i++)
  {
    int a_row_idx = i / b.cols;
    int b_col_idx = i % b.cols;
    float sum = 0;

    for (size_t j = 0; j < a.cols; j++)
    {
      float af = MATRIX_ELEM_AT(a, a_row_idx, j);
      float bf = MATRIX_ELEM_AT(b, j, b_col_idx);
      float new = af *bf;
      // printf("i = %d, j = %d, ri = %d, ci = %d \n", i, j, a_row_idx,
      // b_col_idx); printf("%f * %f = %f \n", af, bf, new);
      sum += new;
      // printf("\n");
    }

    // printf("\n");
    // printf("\n");

    MATRIX_ELEM_AT(dst, a_row_idx, b_col_idx) = sum;
  }
}

void matrix_dot_product(Matrix dst, Matrix a, Matrix b)
{
  int axb = a.cols == b.rows;
  int bxa = b.cols == a.rows;

  // We should be able to match the matrices one way or another.
  // Matrix multiplication is not commutative, so order matters.
  assert(axb || bxa);

  if (axb)
  {
    return ordered_matrix_dot_product(dst, a, b);
  }

  if (bxa)
  {
    return ordered_matrix_dot_product(dst, b, a);
  }
}

// Adds matrix a to matrix dst
void matrix_sum(Matrix dst, Matrix a)
{
  assert(dst.rows == a.rows);
  assert(dst.cols == a.cols);

  for (size_t i = 0; i < a.rows; i++)
  {
    for (size_t j = 0; j < a.cols; j++)
    {
      MATRIX_ELEM_AT(dst, i, j) += MATRIX_ELEM_AT(a, i, j);
    }
  }
}

void matrix_randomise_int(Matrix m, int lb, int ub)
{
  for (size_t i = 0; i < m.rows; i++)
  {
    for (size_t j = 0; j < m.cols; j++)
    {
      MATRIX_ELEM_AT(m, i, j) = random_int(lb, ub);
    }
  }
}

void matrix_randomise(Matrix m, float lb, float ub)
{
  for (size_t i = 0; i < m.rows; i++)
  {
    for (size_t j = 0; j < m.cols; j++)
    {
      MATRIX_ELEM_AT(m, i, j) = random_float(lb, ub);
    }
  }
}

void print_matrix(Matrix m)
{
  int width;

  /* compute the required width */
  for (size_t i = 0; i < m.rows; i++)
  {
    for (size_t j = 0; j < m.cols; j++)
    {
      int w = snprintf(NULL, 0, "%f", MATRIX_ELEM_AT(m, i, j));
      if (width < w)
      {
        width = w;
      }
    }
  }

  /* print the arrays */
  for (size_t i = 0; i < m.rows; i++)
  {
    printf("[");
    for (size_t j = 0; j < m.cols; j++)
    {
      if (j != 0)
        printf(", ");
      printf("%*f", width, MATRIX_ELEM_AT(m, i, j));
    }
    printf("]\n");
  }
}
