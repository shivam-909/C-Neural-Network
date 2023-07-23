#include "cnn.h"
#include <assert.h>
#include <stddef.h>

float diff(float a, float b)
{
  if (a > b)
  {
    return a - b;
  }

  return b - a;
}

float cost(TrainingData *data, int n, Network network)
{
  float total = 0.0f;
  for (size_t i = 0; i < n; i++)
  {
    float sum = 0.0f;
    Matrix input_matrix = data[i].input;
    Matrix output_matrix = data[i].output;

    assert(input_matrix.cols == output_matrix.cols);
    assert(input_matrix.rows == output_matrix.rows);

    Matrix result = feed_forward(input_matrix, network);

    assert(result.cols == output_matrix.cols);
    assert(result.rows == output_matrix.rows);

    for (size_t j = 0; j < result.rows; j++)
    {
      for (size_t k = 0; k < result.cols; k++)
      {
        sum += diff(MATRIX_ELEM_AT(result, j, k),
                    MATRIX_ELEM_AT(output_matrix, j, k));
      }
    }

    total += sum * sum;
  }

  return total / n;
}
