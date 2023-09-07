#include "t1m/t1m.hpp"
#include "t1m/block_scatter_matrix.hpp"
#include "t1m/gemm.hpp"
#include "t1m/index_bundle_finder.hpp"
#include "t1m/scatter_matrix.hpp"
#include "t1m/tensor.hpp"
#include <complex>
#include <memory>
#include <string_view>

namespace t1m
{
  template <typename T>
  struct tensor_type
  {
    using value_type = T;
    using is_complex = std::false_type;
    using is_double = std::is_same<T, double>;
  };

  template <typename T>
  struct tensor_type<std::complex<T>>
  {
    using value_type = T;
    using is_complex = std::true_type;
    using is_double = std::is_same<T, double>;
  };

  template <typename T>
  void contract_internal(Tensor<T> &A, std::vector<int> labelsA,
                         Tensor<T> &B, std::vector<int> labelsB,
                         Tensor<T> &C, std::vector<int> labelsC)
  {
    using TensorType = tensor_type<T>;
    using BaseType = typename TensorType::value_type;
    constexpr bool is_complex = TensorType::is_complex::value;
    constexpr bool is_double = TensorType::is_double::value;

    const cntx_t *cntx = bli_gks_query_cntx();

    const auto BLIS_TYPE = is_double ? BLIS_DOUBLE : BLIS_FLOAT;
    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_TYPE, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_TYPE, BLIS_MR, cntx);
    const dim_t KP = 4;

    const auto ilf = std::make_unique<internal::IndexBundleFinder>(std::move(labelsA), std::move(labelsB), std::move(labelsC));

    auto scatterA = std::make_unique<internal::BlockScatterMatrix<T>>(A, ilf->I, ilf->Pa, MR, KP);
    auto scatterB = std::make_unique<internal::BlockScatterMatrix<T>>(B, ilf->Pb, ilf->J, KP, NR);
    auto scatterC = std::make_unique<internal::BlockScatterMatrix<T>>(C, ilf->Ic, ilf->Jc, MR, NR);

    BaseType alpha = 1.0;
    BaseType beta = 0.0;

    using ContextType = std::conditional_t<is_complex, internal::gemm_context_1m<T, BaseType>, internal::gemm_context<T>>;
    ContextType gemm_ctx = {
        .cntx = cntx,
        .NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_TYPE, BLIS_NC, cntx),
        .KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_TYPE, BLIS_KC, cntx),
        .MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_TYPE, BLIS_MC, cntx),
        .NR = NR,
        .MR = MR,
        .KP = KP,
        .A = scatterA.get(),
        .B = scatterB.get(),
        .C = scatterC.get(),
        .alpha = &alpha,
        .beta = &beta};

    // Choose the correct kernel
    if constexpr (is_double)
    {
      gemm_ctx.kernel = bli_dgemm_ukernel;
    }
    else
    {
      gemm_ctx.kernel = bli_sgemm_ukernel;
    }

    // Run the matrix multiplication
    if constexpr (is_complex)
    {
      internal::gemm_1m(&gemm_ctx);
    }
    else
    {
      internal::gemm(&gemm_ctx);
    }
  }

  std::vector<int> parse_labels(std::string_view labels)
  {
    std::vector<int> res;
    res.reserve(labels.size());
    for (const auto &c : labels)
    {
      res.push_back(static_cast<int>(c));
    }
    return res;
  }

  template <typename T>
  void contract_internal(Tensor<T> &A, const std::string_view labelsA,
                         Tensor<T> &B, const std::string_view labelsB,
                         Tensor<T> &C, const std::string_view labelsC)
  {
    contract_internal(A, parse_labels(labelsA), B, parse_labels(labelsB), C, parse_labels(labelsC));
  }

  template <>
  void contract(Tensor<std::complex<float>> &A, const std::string_view labelsA,
                Tensor<std::complex<float>> &B, const std::string_view labelsB,
                Tensor<std::complex<float>> &C, const std::string_view labelsC)
  {
    contract_internal(A, labelsA, B, labelsB, C, labelsC);
  }

  template <>
  void contract(Tensor<std::complex<double>> &A, const std::string_view labelsA,
                Tensor<std::complex<double>> &B, const std::string_view labelsB,
                Tensor<std::complex<double>> &C, const std::string_view labelsC)
  {
    contract_internal(A, labelsA, B, labelsB, C, labelsC);
  }

  template <>
  void contract(Tensor<float> &A, const std::string_view labelsA,
                Tensor<float> &B, const std::string_view labelsB,
                Tensor<float> &C, const std::string_view labelsC)
  {
    contract_internal(A, labelsA, B, labelsB, C, labelsC);
  }

  template <>
  void contract(Tensor<double> &A, const std::string_view labelsA,
                Tensor<double> &B, const std::string_view labelsB,
                Tensor<double> &C, const std::string_view labelsC)
  {
    contract_internal(A, labelsA, B, labelsB, C, labelsC);
  }
};
