#include "cnn.h"

Layer new_layer(Matrix neurons, Matrix biases, enum ACTIVATION_FUNCTION af)
{
  Layer layer;

  layer.neurons = neurons;
  layer.biases = biases;
  layer.af = af;

  return layer;
}

Layer construct_layer(int nn, int ni, int lb, int ub,
                      enum ACTIVATION_FUNCTION af)
{
  Matrix m = new_matrix(ni, nn);
  matrix_randomise(m, lb, ub);

  Matrix b = new_matrix(1, nn);
  matrix_randomise(b, lb, ub);

  Layer layer;

  layer.neurons = m;
  layer.biases = b;
  layer.af = af;

  return layer;
}

void feed_layer(Matrix dst, Matrix input, Layer layer)
{
  matrix_dot_product(dst, input, layer.neurons);
  matrix_sum(dst, layer.biases);
  sigmoid_matrix(dst);
}

void print_layer(Layer layer, const char *name)
{
  print_matrix(layer.neurons, name);
  print_matrix(layer.biases, name);
}
