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

void free_matrix(Matrix m);

void copy_matrix(Matrix dst, Matrix src);

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

// ACTIVATION FUNCTIONS ----------------------------------------------------

enum ACTIVATION_FUNCTION
{
  linear,
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

void free_layer(Layer layer);

Layer construct_layer(int nn, int ni, int lb, int ub,
                      enum ACTIVATION_FUNCTION af);

void feed_layer(Matrix dst, Matrix input, Layer layer);

void print_layer(Layer layer);

// NETWORK -----------------------------------------------------------------

typedef struct
{
  Layer *layers;
  size_t size;
} Network;

Network new_network(int n, ...);

void free_network(Network n);

void feed_forward(Matrix dst, Matrix input, Network network);

void print_network(Network network, const char *name);

// TRAINING ----------------------------------------------------------------

typedef struct
{
  Matrix input;
  Matrix output;
} TrainingData;

typedef struct
{
  TrainingData *td;
  size_t size;
} TrainingDataCollection;

float cost(TrainingDataCollection td, Network network);

void learn(Network dst, Network network, TrainingDataCollection td, float eps);

void train(Network gradients, Network network, TrainingDataCollection td,
           size_t iterations, float eps, float learn_rate);

// UTILITIES ---------------------------------------------------------------

#define MATRIX_PRINT(m) print_matrix(m, #m)
#define MATRIX_ELEM_AT(m, i, j) ((m).es[(i) * (m).cols + (j)])
#define NETWORK_PRINT(n) print_network(n, #n)

// Returns a random float between lb and ub.
float random_float(float lb, float ub);

// Returns a random integer between lb and ub.
int random_int(int lb, int ub);

#endif // CNN_H
