#pragma once

#include <complex>
#include "tensor.hpp"
#include "utils.hpp"

namespace t1m
{
  void contract(Tensor<std::complex<float>> A, std::string labelsA,
                Tensor<std::complex<float>> B, std::string labelsB,
                Tensor<std::complex<float>> C, std::string labelsC);

  void contract(Tensor<std::complex<double>> A, std::string labelsA,
                Tensor<std::complex<double>> B, std::string labelsB,
                Tensor<std::complex<double>> C, std::string labelsC);

  void contract(Tensor<float> A, std::string labelsA,
                Tensor<float> B, std::string labelsB,
                Tensor<float> C, std::string labelsC);

  void contract(Tensor<double> A, std::string labelsA,
                Tensor<double> B, std::string labelsB,
                Tensor<double> C, std::string labelsC);
};
