#include "cnn.h"

float random_float(float lb, float ub)
{
  return ((float)rand() / (float)RAND_MAX) * (ub - lb) + lb;
}

int random_int(int lb, int ub) { return (int)random_float(lb, ub); }
