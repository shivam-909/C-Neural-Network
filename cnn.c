
#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CNN_IMPLEMENTATION

void setup() { srand(time(0)); }

int main()
{
  setup();

  int lb = 1;
  int ub = 50;

  Matrix a = new_matrix(4, 3);

  matrix_randomise_int(a, lb, ub);

  MATRIX_PRINT(a);

  return 0;
}
