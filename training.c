#include "cnn.h"
#include <stddef.h>
#include <stdio.h>

void gradient_approximation(Network network, Matrix dst_mat, Matrix net_mat,
                            TrainingDataCollection td, float fx, float eps)
{
  for (size_t row = 0; row < net_mat.rows; row++)
  {
    for (size_t col = 0; col < net_mat.cols; col++)
    {

      float tmp = MATRIX_ELEM_AT(net_mat, row, col);
      MATRIX_ELEM_AT(net_mat, row, col) += eps;

      float fxh = cost(td, network);
      float dydx = (fxh - fx) / eps;

      MATRIX_ELEM_AT(dst_mat, row, col) = dydx;
      MATRIX_ELEM_AT(net_mat, row, col) = tmp;
    }
  }
}

void learn(Network dst, Network network, TrainingDataCollection td, float eps)
{

  float fx = cost(td, network);

  for (size_t li = 0; li < network.size; li++)
  {

    gradient_approximation(network, dst.layers[li].neurons,
                           network.layers[li].neurons, td, fx, eps);

    gradient_approximation(network, dst.layers[li].biases,
                           network.layers[li].biases, td, fx, eps);
  }
}

void descend(Network gradients, Network network, float learn_rate)
{
  for (size_t li = 0; li < network.size; li++)
  {

    for (size_t row = 0; row < network.layers[li].neurons.rows; row++)
    {
      for (size_t col = 0; col < network.layers[li].neurons.cols; col++)
      {
        MATRIX_ELEM_AT(network.layers[li].neurons, row, col) -=
            (MATRIX_ELEM_AT(gradients.layers[li].neurons, row, col) *
             learn_rate);
      }
    }

    for (size_t row = 0; row < network.layers[li].biases.rows; row++)
    {
      for (size_t col = 0; col < network.layers[li].biases.cols; col++)
      {
        MATRIX_ELEM_AT(network.layers[li].biases, row, col) -=
            (MATRIX_ELEM_AT(gradients.layers[li].biases, row, col) *
             learn_rate);
      }
    }
  }
}

void train(Network gradients, Network network, TrainingDataCollection td,
           size_t iterations, float eps, float learn_rate)
{

  for (size_t i = 0; i < iterations; i++)
  {
    learn(gradients, network, td, eps);
    descend(gradients, network, learn_rate);
  }
}
