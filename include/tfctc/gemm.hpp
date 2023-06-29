#pragma once

#include <complex>
#include "utils.hpp"
#include "std_ext.hpp"
#include "gemm_context.hpp"
#include "scatter_matrix.hpp"
#include "block_scatter_matrix.hpp"
#include "packing.hpp"
#include "packing_1m.hpp"
#include "blis.h"

namespace tfctc
{
  namespace internal
  {
    template <typename T>
    void gemm_1m(const gemm_context_1m<std::complex<T>, T>* gemm_ctx)
    {
      ScatterMatrix<std::complex<T>>* A = gemm_ctx->A;
      ScatterMatrix<std::complex<T>>* B = gemm_ctx->B;
      BlockScatterMatrix<std::complex<T>>* C = gemm_ctx->C;

      const dim_t NC = gemm_ctx->NC;
      const dim_t KC = gemm_ctx->KC;
      const dim_t MC = gemm_ctx->MC;
      const dim_t NR = gemm_ctx->NR;
      const dim_t MR = gemm_ctx->MR;

      const size_t M = A->row_size();
      const size_t K = A->col_size();
      const size_t N = B->col_size();

      size_t I, J, x, y;
      dim_t m1, n1, k1, m, n;
      inc_t rsc = 1, csc;

      T* buf = nullptr;
      T* A_tilde = nullptr; // A in G^{MC x KC}
      T* A_tilde_base = nullptr;

      T* B_tilde = nullptr; // B in G^{KC x NC}
      T* B_tilde_base = nullptr;

      T* C_tilde = nullptr; // C in G^{MC x NC}

      tfctc::utils::alloc_aligned<T>(&buf, MC * KC + KC * NC + MC * NC);

      A_tilde = A_tilde_base = buf;
      B_tilde = B_tilde_base = buf + MC * KC;
      C_tilde = buf + MC * KC + KC * NC;

      for (size_t j_c = 0; j_c < N; j_c += NC)
      {
        J = j_c / NC;
        for (size_t p_c = 0; p_c < K; p_c += KC / 2)
        {
          k1 = tfctc::std_ext::min(KC / 2, static_cast<dim_t>(K - p_c));
          n1 = tfctc::std_ext::min(NC, static_cast<dim_t>(N - j_c));

          memset(B_tilde, 0, KC * NC * sizeof(T));
          gemm_ctx->pack_B(B, B_tilde, p_c, j_c, k1, n1, NR);

          // B is now row-major packed into a KC * NC buffer
          // with the specialized format such that each sliver
          // has stride NR

          for (size_t i_c = 0; i_c < M; i_c += MC / 2)
          {
            I = i_c / MC / 2;
            m1 = tfctc::std_ext::min(MC / 2, static_cast<dim_t>(M - i_c));

            memset(A_tilde, 0, MC * KC * sizeof(T));
            gemm_ctx->pack_A(A, A_tilde, i_c, p_c, m1, k1, MR);

            // A is now column-major packed into a MC * KC buffer
            // with the specialized format such that each sliver
            // has stride MR

            // Now treat everything as real-valued:
            // Multiplication by two since: 1 complex = 2 real
            // Use NR, MR as with real-valued mm
            m1 = tfctc::std_ext::min(MC, static_cast<dim_t>(M - i_c) * 2);
            k1 = tfctc::std_ext::min(KC, static_cast<dim_t>(K - p_c) * 2);

            for (size_t j_r = 0; j_r < n1; j_r += NR)
            {
              n = tfctc::std_ext::min(NR, static_cast<dim_t>(n1 - j_r));

              for (size_t i_r = 0; i_r < m1; i_r += MR)
              {
                m = tfctc::std_ext::min(MR, static_cast<dim_t>(m1 - i_r));

                // Find strides for current I, J
                // Divide stride by two since: 1 complex = 2 real
                // Use row and column strides as BLIS-Kernel parameters
                rsc = C->row_stride_in_block(I + i_r / MR / 2) / 2;
                csc = C->col_stride_in_block(J + j_r / NR) / 2;

                x = i_c + (i_r / 2);
                y = j_c + j_r;

                if (rsc > 0 && csc > 0)
                {
                  gemm_ctx->kernel(m, n, k1,
                    gemm_ctx->alpha,
                    A_tilde,
                    B_tilde,
                    gemm_ctx->beta,
                    reinterpret_cast<T*>(C->pointer_at_loc(x, y)), rsc, csc,
                    nullptr,
                    gemm_ctx->cntx);
                }
                else {
                  gemm_ctx->kernel(m, n, k1,
                    gemm_ctx->alpha,
                    A_tilde,
                    B_tilde,
                    gemm_ctx->beta,
                    C_tilde, 1, m,
                    nullptr,
                    gemm_ctx->cntx);

                  gemm_ctx->unpack_C(C, C_tilde, x, y, m, n);
                }

                A_tilde += MR * k1;
              }
              B_tilde += k1 * NR;

              A_tilde = A_tilde_base;
            }
            B_tilde = B_tilde_base;
          }
        }
      }

      free(buf);
    }

    template <typename T>
    void gemm(const gemm_context<T>* gemm_ctx)
    {
      BlockScatterMatrix<T>* A = gemm_ctx->A;
      BlockScatterMatrix<T>* B = gemm_ctx->B;
      BlockScatterMatrix<T>* C = gemm_ctx->C;

      const dim_t NC = gemm_ctx->NC;
      const dim_t KC = gemm_ctx->KC;
      const dim_t MC = gemm_ctx->MC;
      const dim_t NR = gemm_ctx->NR;
      const dim_t MR = gemm_ctx->MR;

      const size_t M = A->row_size();
      const size_t K = A->col_size();
      const size_t N = B->col_size();

      T* buf = nullptr;
      T* A_tilde = nullptr; // A in G^{MC x KC}
      T* A_tilde_base = nullptr;

      T* B_tilde = nullptr; // B in G^{KC x NC}
      T* B_tilde_base = nullptr;

      T* C_tilde = nullptr; // C in G^{MC x NC}
      T* C_tilde_base = nullptr;

      tfctc::utils::alloc_aligned<T>(&buf, MC * KC + KC * NC + MC * NC);

      A_tilde = A_tilde_base = buf;
      B_tilde = B_tilde_base = buf + MC * KC;
      C_tilde = C_tilde_base = buf + MC * KC + KC * NC;

      size_t x, y;
      dim_t m1, n1, k1, m, n;
      inc_t rsc = 1, csc;

      for (size_t j_c = 0; j_c < N; j_c += NC)
      {
        n1 = tfctc::std_ext::min(NC, static_cast<dim_t>(N - j_c));

        for (size_t p_c = 0; p_c < K; p_c += KC)
        {
          k1 = tfctc::std_ext::min(KC, static_cast<dim_t>(K - p_c));

          memset(B_tilde, 0, KC * NC * sizeof(T));
          internal::pack_b(B, B_tilde, p_c, j_c, k1, n1, NR, 4);

          for (size_t i_c = 0; i_c < M; i_c += MC)
          {
            m1 = tfctc::std_ext::min(MC, static_cast<dim_t>(M - i_c));

            memset(A_tilde, 0, MC * KC * sizeof(T));
            internal::pack_a(A, A_tilde, i_c, p_c, m1, k1, MR, 4);

            for (size_t j_r = 0; j_r < n1; j_r += NR)
            {
              n = tfctc::std_ext::min(NR, static_cast<dim_t>(n1 - j_r));
              
              for (size_t i_r = 0; i_r < m1; i_r += MR)
              {
                m = tfctc::std_ext::min(MR, static_cast<dim_t>(m1 - i_r));

                rsc = C->row_stride_in_block(i_c / MC + i_r / MR);
                csc = C->col_stride_in_block(j_c / NC + j_r / NR);

                x = i_c + i_r;
                y = j_c + j_r;

                if (rsc > 0 && csc > 0)
                {
                  gemm_ctx->kernel(m, n, k1,
                    gemm_ctx->alpha,
                    A_tilde,
                    B_tilde,
                    gemm_ctx->beta,
                    C->pointer_at_loc(x, y), rsc, csc,
                    nullptr,
                    gemm_ctx->cntx);
                }
                else {
                  gemm_ctx->kernel(m, n, k1,
                    gemm_ctx->alpha,
                    A_tilde,
                    B_tilde,
                    gemm_ctx->beta,
                    C_tilde, 1, m,
                    nullptr,
                    gemm_ctx->cntx);

                  internal::unpack_c_scat(C, C_tilde, x, y, m, n);
                }

                A_tilde += MR * k1;
              }
              B_tilde += k1 * NR;

              A_tilde = A_tilde_base;
            }
            B_tilde = B_tilde_base;
          }
        }
      }

      free(buf);
    }
  }
};
