#include "t1m/t1m_c.h"
#include "t1m/t1m.hpp"
#include <complex>
#include <vector>

extern "C" {
    void t1m_contract(const _Complex double *A, const size_t *shapeA, int ndimA, const int *labelsA,
                    const _Complex double *B, const size_t *shapeB, int ndimB, const int *labelsB,
                    _Complex double *C, const size_t *shapeC, int ndimC, const int *labelsC)
    {
        std::vector<size_t> shapeA_(shapeA, shapeA + ndimA);
        std::vector<size_t> shapeB_(shapeB, shapeB + ndimB);
        std::vector<size_t> shapeC_(shapeC, shapeC + ndimC);

        std::vector<int> labelsA_(labelsA, labelsA + ndimA);
        std::vector<int> labelsB_(labelsB, labelsB + ndimB);
        std::vector<int> labelsC_(labelsC, labelsC + ndimC);

        // Cast away const because I don't know how to go from const pointer directly
        // to const std::complex<double>
        auto Araw = const_cast<_Complex double *>(A);
        auto Braw = const_cast<_Complex double *>(B);

        // Cast from C complex to C++ complex
        const auto Acpp = reinterpret_cast<std::complex<double> *>(Araw);
        const auto Bcpp = reinterpret_cast<std::complex<double> *>(Braw);
        auto Ccpp = reinterpret_cast<std::complex<double> *>(C);

        auto A_ = t1m::Tensor<std::complex<double>>(shapeA_, Acpp);
        auto B_ = t1m::Tensor<std::complex<double>>(shapeB_, Bcpp);
        auto C_ = t1m::Tensor<std::complex<double>>(shapeC_, Ccpp);

        t1m::contract(A_, std::move(labelsA_),
                    B_, std::move(labelsB_),
                    C_, std::move(labelsC_));
    }
}