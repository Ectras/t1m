#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void t1m_contract(const _Complex double *A, const size_t *shapeA, int ndimA, const int *labelsA,
                  const _Complex double *B, const size_t *shapeB, int ndimB, const int *labelsB,
                  _Complex double *C, const size_t *shapeC, int ndimC, const int *labelsC);

#ifdef __cplusplus
}
#endif