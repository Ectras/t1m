#include "t1m/t1m_c.h"
#include "t1m/t1m.hpp"
#include <complex>
#include <vector>

extern "C" {
    void t1m_contract(_Complex double *A, size_t *shapeA, int ndimA, char *labelsA,
                    _Complex double *B, size_t *shapeB, int ndimB, char *labelsB,
                    _Complex double *C, size_t *shapeC, int ndimC, char *labelsC)
    {
        std::vector<size_t> shapeA_(shapeA, shapeA + ndimA);
        std::vector<size_t> shapeB_(shapeB, shapeB + ndimB);
        std::vector<size_t> shapeC_(shapeC, shapeC + ndimC);

        std::vector<char> labelsA_(labelsA, labelsA + ndimA);
        std::vector<char> labelsB_(labelsB, labelsB + ndimB);
        std::vector<char> labelsC_(labelsC, labelsC + ndimC);

        auto Acpp = reinterpret_cast<std::complex<double> *>(A);
        auto Bcpp = reinterpret_cast<std::complex<double> *>(B);
        auto Ccpp = reinterpret_cast<std::complex<double> *>(C);

        auto A_ = t1m::Tensor<std::complex<double>>(shapeA_, Acpp);
        auto B_ = t1m::Tensor<std::complex<double>>(shapeB_, Bcpp);
        auto C_ = t1m::Tensor<std::complex<double>>(shapeC_, Ccpp);

        t1m::contract(A_, std::string(labelsA_.begin(), labelsA_.end()),
                    B_, std::string(labelsB_.begin(), labelsB_.end()),
                    C_, std::string(labelsC_.begin(), labelsC_.end()));
    }
}