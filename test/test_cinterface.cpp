#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "t1m/t1m_c.h"
#include <complex.h>
#include <stdlib.h>

using DoubleComplex = _Complex double;

template <typename T>
void alloc_aligned(T **ptr, size_t n)
{
    if (posix_memalign((void **)ptr, 32, n * sizeof(T)))
    {
        std::throw_with_nested(std::bad_alloc());
    }
}

inline void require(DoubleComplex a, DoubleComplex b)
{
  REQUIRE(creal(a) == doctest::Approx(creal(b)).epsilon(0.001));
  REQUIRE(cimag(a) == doctest::Approx(cimag(b)).epsilon(0.001));
}

inline void requireAll(DoubleComplex *tensor, std::vector<DoubleComplex> expected)
{
  for(int i = 0; i < expected.size(); i++)
  {
    require(tensor[i], expected[i]);
  }
}

TEST_CASE("(double) 2D . 2D => 2D")
{
    DoubleComplex *A = nullptr, *B = nullptr, *C = nullptr;
    alloc_aligned(&A, 2 * 2);
    alloc_aligned(&B, 2 * 2);
    alloc_aligned(&C, 2 * 2);

    A[0] = 3. + 2. * I;
    A[1] = 0. - 1. * I;
    A[2] = 0. + 1. * I;
    A[3] = 1. + 0. * I;

    B[0] = 4.0 + 0.0 * I;
    B[1] = -0.5 + 0.5 * I;
    B[2] = 0. + 7. * I;
    B[3] = 3.3 + 0. * I;

    // Output tensor must be zero-initialized
    memset(C, 0, 2 * 2 * sizeof(DoubleComplex));

    size_t shapeA[] = {2, 2};
    size_t shapeB[] = {2, 2};
    size_t shapeC[] = {2, 2};
    

    SUBCASE("standard")
    {
        t1m_contract(A, shapeA, 2, "ab", B, shapeB, 2, "bc", C, shapeC, 2, "ac");

        requireAll(C, {11.5 + 7.5 * I,
                       -0.5 -3.5 * I,
                       -14 + 24.3 * I,
                       10.3 + 0. * I});
    }
}