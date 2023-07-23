
#include "cnn.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CNN_IMPLEMENTATION

void setup() { srand(time(0)); }

// 3a * 3b
void training_data(TrainingData *td, int n)
{
  for (size_t i = 0; i < n; i++)
  {
    int a = random_int(0, 10);
    int b = random_int(0, 10);

    Matrix in = new_matrix(1, 2);

    MATRIX_ELEM_AT(in, 0, 1) = a;
    MATRIX_ELEM_AT(in, 0, 2) = b;

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
  Layer l1 = construct_layer(2, 2, 0, 1, sig);
  // 1 Neuron layer, accepts 2 inputs
  Layer l2 = construct_layer(1, 2, 0, 1, sig);

  Network network = new_network(2, l1, l2);

  MATRIX_PRINT(network.layers[0].neurons);
  MATRIX_PRINT(network.layers[1].neurons);
  MATRIX_PRINT(network.layers[0].biases);
  MATRIX_PRINT(network.layers[1].biases);

  Matrix result = feed_forward(x, network);

  MATRIX_PRINT(result);

  int n = 10;
  TrainingData *td = malloc(sizeof(TrainingData) * n);

  training_data(td, n);

  float network_cost = cost(td, n, network);

  printf("%f", network_cost);

  train(network, td, n, 1000, 1e-3, 1e-3);

  float new_cost = cost(td, n, network);

  printf("%f", new_cost);
}

int main_manual()
{
  setup();

  int lb = 0;
  int ub = 1;

  // Input matrix
  // [ a, b ]
  Matrix x = new_matrix(1, 2);

  printf("FIRST LAYER \n");

  // First layer, each column is a neuron.
  // [ w1, w2 ]
  // [ w3, w4 ]
  Matrix l1 = new_matrix(2, 2);

  // Bias
  Matrix b1 = new_matrix(1, 2);

  matrix_randomise(x, lb, ub);
  matrix_randomise(l1, lb, ub);
  matrix_randomise(b1, lb, ub);

  MATRIX_PRINT(x);
  MATRIX_PRINT(l1);
  MATRIX_PRINT(b1);

  // Capture intermediate result.
  Matrix r1 = new_matrix(1, 2);

  // [ a, b ] * [ w1, w2 ] == [ a*w1 + a*w3, b*w2 + b*w4]
  //            [ w3, w4 ]
  matrix_dot_product(r1, x, l1);

  MATRIX_PRINT(r1);

  // Add bias
  matrix_sum(r1, b1);

  // Apply activation function
  sigmoid_matrix(r1);

  // r1 now contains the result of the first layer.
  MATRIX_PRINT(r1);

  printf("\nSECOND LAYER \n");

  // Another layer, single neuron.
  // [ w1 ]
  // [ w2 ]
  Matrix l2 = new_matrix(2, 1);
  Matrix b2 = new_matrix(1, 1);

  matrix_randomise(l2, lb, ub);
  matrix_randomise(b2, lb, ub);

  MATRIX_PRINT(l2);
  MATRIX_PRINT(b2);

  Matrix r2 = new_matrix(1, 1);

  // Feed result from last layer into next layer.
  matrix_dot_product(r2, r1, l2);

  MATRIX_PRINT(r2);

  matrix_sum(r2, b2);

  sigmoid_matrix(r2);

  MATRIX_PRINT(r2);

  return 0;
}
