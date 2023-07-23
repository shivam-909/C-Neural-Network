#include "cnn.h"
#include <stddef.h>

void finite_diff(Network network, TrainingData *training_input, int n,
                 float eps, float learn_rate)
{
  for (size_t layer_idx = 0; layer_idx < network.n; layer_idx++)
  {
    Layer l = network.layers[layer_idx];
    for (size_t row = 0; row < l.neurons.rows; row++)
    {
      for (size_t col = 0; col < l.neurons.rows; col++)
      {
        float fx = cost(training_input, n, network);
        float tmp = MATRIX_ELEM_AT(l.neurons, row, col);
        MATRIX_ELEM_AT(l.neurons, row, col) = tmp + (eps * learn_rate);
        float fxh = cost(training_input, n, network);
        MATRIX_ELEM_AT(l.neurons, row, col) = tmp - (fxh - fx) / eps;
      }
    }

    for (size_t row = 0; row < l.biases.rows; row++)
    {
      for (size_t col = 0; col < l.biases.rows; col++)
      {
        float fx = cost(training_input, n, network);
        float tmp = MATRIX_ELEM_AT(l.biases, row, col);
        MATRIX_ELEM_AT(l.biases, row, col) = tmp + (eps * learn_rate);
        float fxh = cost(training_input, n, network);
        MATRIX_ELEM_AT(l.biases, row, col) = tmp - (fxh - fx) / eps;
      }
    }
  }
}

void train(Network network, TrainingData *training_data, int n,
           float iterations, float eps, float learn_rate)
{
  for (size_t i = 0; i < iterations; i++)
  {
    finite_diff(network, training_data, n, eps, learn_rate);
  }
}
