#ifndef CNN_H
#define CNN_H

#include <stdlib.h>

// MATRICES ----------------------------------------------------------------

typedef struct
{
  size_t rows;
  size_t cols;
  float *es;
} Matrix;

// Allocates a new matrix of size rows x cols;
Matrix new_matrix(size_t rows, size_t cols);

// Multiplies matrices a and b and stores the result in dst.
// Asserts that a.cols == b.rows and dst.cols == a.cols and dst.rows == b.rows.
void matrix_dot_product(Matrix dst, Matrix a, Matrix b);

// Adds matrix a to matrix dst.
void matrix_sum(Matrix dst, Matrix a);

// Initialises a matrix with random values between lb and ub.
void matrix_randomise(Matrix m, float lb, float ub);

// Initialises a matrix with random integer values.
// Note that the data is still an array of floats but rounded to the nearest
// integer. Useful for inspection of logic.
void matrix_randomise_int(Matrix m, int lb, int ub);

// Prints a matrix.
void print_matrix(Matrix m, const char *name);

// UTILITIES ---------------------------------------------------------------

#define MATRIX_PRINT(m) print_matrix(m, #m)
#define MATRIX_ELEM_AT(m, i, j) ((m).es[(i) * (m).cols + (j)])

// Returns a random float between lb and ub.
float random_float(float lb, float ub);

// Returns a random integer between lb and ub.
int random_int(int lb, int ub);

// ACTIVATION FUNCTIONS ----------------------------------------------------

enum ACTIVATION_FUNCTION
{
  sig
};

float sigmoid(float x);

void sigmoid_matrix(Matrix m);

// LAYER -------------------------------------------------------------------

typedef struct
{
  // Defines the neurons in the layer, one column per neuron.
  // Each cell represents the weight of a synapse into that neuron.
  Matrix neurons;

  // Biases.
  // Must be one bias for each neuron in the layer.
  Matrix biases;

  // Activation functions.
  // Must be one activation function for each neuron in the layer.
  enum ACTIVATION_FUNCTION af;

} Layer;

Layer new_layer(Matrix neurons, Matrix biases, enum ACTIVATION_FUNCTION af);

Layer construct_layer(int nn, int ni, int lb, int ub,
                      enum ACTIVATION_FUNCTION af);

void feed_layer(Matrix dst, Matrix input, Layer layer);

void print_layer(Layer layer, const char *name);

// NETWORK -----------------------------------------------------------------

typedef struct
{
  Layer *layers;
  int n;
} Network;

Network new_network(int n, ...);

Matrix feed_forward(Matrix input, Network network);

#endif // CNN_H
