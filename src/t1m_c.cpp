#include "t1m/t1m_c.h"
#include "t1m/t1m.hpp"
#include <complex>
#include <vector>

extern "C" {
    void t1m_contract(_Complex double *A, const size_t *shapeA, int ndimA, const int *labelsA,
                    _Complex double *B, const size_t *shapeB, int ndimB, const int *labelsB,
                    _Complex double *C, const size_t *shapeC, int ndimC, const int *labelsC)
    {
        std::vector<size_t> shapeA_(shapeA, shapeA + ndimA);
        std::vector<size_t> shapeB_(shapeB, shapeB + ndimB);
        std::vector<size_t> shapeC_(shapeC, shapeC + ndimC);

        std::vector<int> labelsA_(labelsA, labelsA + ndimA);
        std::vector<int> labelsB_(labelsB, labelsB + ndimB);
        std::vector<int> labelsC_(labelsC, labelsC + ndimC);

        auto Acpp = reinterpret_cast<std::complex<double> *>(A);
        auto Bcpp = reinterpret_cast<std::complex<double> *>(B);
        auto Ccpp = reinterpret_cast<std::complex<double> *>(C);

        auto A_ = t1m::Tensor<std::complex<double>>(shapeA_, Acpp);
        auto B_ = t1m::Tensor<std::complex<double>>(shapeB_, Bcpp);
        auto C_ = t1m::Tensor<std::complex<double>>(shapeC_, Ccpp);

        t1m::contract(A_, std::move(labelsA_),
                    B_, std::move(labelsB_),
                    C_, std::move(labelsC_));
    }
}