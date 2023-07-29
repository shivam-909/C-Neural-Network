#include "cnn.h"
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

Network new_network(int n, ...)
{

  Network network;
  network.layers = malloc(sizeof(Layer) * n);
  network.size = n;

  va_list args;
  va_start(args, n);

  for (size_t i = 0; i < (size_t)n; i++)
  {
    Layer layer = va_arg(args, Layer);
    network.layers[i] = layer;
  }

  return network;
}

void free_network(Network n)
{
  for (size_t i = 0; i < n.size; i++)
  {
    free_layer(n.layers[i]);
  }
}

void feed_forward(Matrix dst, Matrix input, Network network)
{
  Matrix ic = new_matrix(input.rows, input.cols);
  copy_matrix(ic, input);

  for (size_t i = 0; i < network.size; i++)
  {
    Matrix ir = new_matrix(ic.rows, network.layers[i].neurons.cols);
    feed_layer(ir, ic, network.layers[i]);
    free_matrix(ic);
    ic = ir;
  }

  copy_matrix(dst, ic);
  free_matrix(ic);
}

void print_network(Network network, const char *name)
{
  printf("----------------------------------------------------------------\n");
  printf("%s: \n", name);
  for (size_t i = 0; i < network.size; i++)
  {
    printf("--------------------------------\n");
    printf("Layer: %zu, Network: %s \n", i, name);
    print_layer(network.layers[i]);
    printf("--------------------------------\n");
  }
  printf("----------------------------------------------------------------\n");
}
