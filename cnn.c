
#include "cnn.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CNN_IMPLEMENTATION

void setup() { srand(time(0)); }

// 3a + 3b
void training_data(TrainingData *td, int n)
{
  for (size_t i = 0; i < n; i++)
  {
    int a = random_int(0, 10);
    int b = random_int(0, 10);

    Matrix in = new_matrix(1, 2);

    MATRIX_ELEM_AT(in, 0, 0) = a;
    MATRIX_ELEM_AT(in, 0, 1) = b;

    Matrix out = new_matrix(1, 1);

    MATRIX_ELEM_AT(out, 0, 0) = (3 * a) + (3 * b);

    TrainingData ntd;

    ntd.input = in;
    ntd.output = out;
    td[i] = ntd;
  }
}

int main()
{
  setup();

  // 2 inputs
  Matrix x = new_matrix(1, 2);
  matrix_randomise_int(x, 0, 5);
  MATRIX_PRINT(x);

  // 2 Neuron layer
  Layer l1 = construct_layer(2, 2, 0, 5, sig);
  // 1 Neuron layer, accepts 2 inputs
  Layer l2 = construct_layer(1, 2, 0, 5, sig);

  Network network = new_network(2, l1, l2);
  NETWORK_PRINT(network);

  int n = 20;
  TrainingData *td = malloc(sizeof(TrainingData) * n);
  training_data(td, n);

  TrainingDataCollection tdc;
  tdc.td = td;
  tdc.size = n;

  float network_cost = cost(tdc, network);

  Layer g1 = construct_layer(2, 2, 0, 1, sig);
  Layer g2 = construct_layer(1, 2, 0, 1, sig);
  Network gradients = new_network(2, g1, g2);

  train(gradients, network, tdc, 2000000, 1e-5, 1e-5);

  float new_cost = cost(tdc, network);

  printf("Network cost before training: %f\n", network_cost);
  printf("Network cost after training %f\n", new_cost);

  Matrix result = new_matrix(1, 1);
  feed_forward(result, x, network);
  MATRIX_PRINT(result);

  free_network(network);
  free_network(gradients);
  free_matrix(x);
  free_matrix(result);
  free(td);
}
