
#include <stdlib.h>
#define CNN_IMPLEMENTATION
#include "cnn.h"
#include <stdio.h>
#include <time.h>

void setup() { srand(time(0)); }

int main()
{
  setup();

  int lb = 1;
  int ub = 50;

  Matrix a = new_matrix(4, 3);
  Matrix b = new_matrix(3, 6);
  Matrix c = new_matrix(4, 6);

  matrix_randomise_int(a, lb, ub);
  matrix_randomise_int(b, lb, ub);

  print_matrix(a);

  printf("\n");

  print_matrix(b);

  printf("\n");
  printf("\n");

  matrix_dot_product(c, a, b);

  printf("\n");
  printf("\n");

  print_matrix(c);

  return 0;
}
