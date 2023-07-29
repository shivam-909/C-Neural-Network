#include "cnn.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>

float diff(float a, float b)
{
  if (a > b)
  {
    return a - b;
  }

  return b - a;
}

float cost(TrainingDataCollection td, Network network)
{
  float total = 0.0f;

  for (size_t i = 0; i < td.size; i++)
  {
    float sum = 0.0f;

    Matrix input_matrix = td.td[i].input;
    Matrix output_matrix = td.td[i].output;

    Matrix result = new_matrix(output_matrix.rows, output_matrix.cols);
    feed_forward(result, input_matrix, network);

    assert(result.cols == output_matrix.cols);
    assert(result.rows == output_matrix.rows);

    for (size_t j = 0; j < result.rows; j++)
    {
      for (size_t k = 0; k < result.cols; k++)
      {
        float d = diff(MATRIX_ELEM_AT(result, j, k),
                       MATRIX_ELEM_AT(output_matrix, j, k));

        sum += (d * d);
      }
    }

    free_matrix(result);

    total += (sqrt(sum));
  }

  return total / td.size;
}
