#include "cnn.h"
#include <math.h>
#include <stddef.h>

float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

void sigmoid_matrix(Matrix m)
{
  for (size_t i = 0; i < m.rows; i++)
  {
    for (size_t j = 0; j < m.cols; j++)
    {
      MATRIX_ELEM_AT(m, i, j) = sigmoid(MATRIX_ELEM_AT(m, i, j));
    }
  }
}
