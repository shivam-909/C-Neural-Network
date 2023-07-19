#include "cnn.h"
#include <stdarg.h>
#include <stddef.h>

Network new_network(int n, ...)
{

  Network network;
  network.layers = malloc(sizeof(Layer) * n);
  network.n = n;

  va_list args;
  va_start(args, n);

  for (size_t i = 0; i < (size_t)n; i++)
  {
    Layer layer = va_arg(args, Layer);
    network.layers[i] = layer;
  }

  return network;
}

Matrix feed_forward(Matrix input, Network network)
{
  Matrix x = input;

  for (size_t i = 0; i < network.n; i++)
  {
    Matrix ir = new_matrix(x.rows, network.layers[i].neurons.cols);
    feed_layer(ir, x, network.layers[i]);
    x = ir;
  }

  return x;
}
