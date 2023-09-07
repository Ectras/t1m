#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void t1m_contract(_Complex double *A, size_t *shapeA, int ndimA, const int *labelsA,
                  _Complex double *B, size_t *shapeB, int ndimB, const int *labelsB,
                  _Complex double *C, size_t *shapeC, int ndimC, const int *labelsC);

#ifdef __cplusplus
}
#endif