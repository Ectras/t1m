#pragma once

#include <complex>
#include <string_view>
#include "tensor.hpp"
#include "utils.hpp"

namespace t1m
{
  template <typename T>
  void contract(Tensor<T> A, const std::string_view labelsA,
                Tensor<T> B, const std::string_view labelsB,
                Tensor<T> C, const std::string_view labelsC) = delete;

  template <>
  void contract(Tensor<std::complex<float>> A, const std::string_view labelsA,
                Tensor<std::complex<float>> B, const std::string_view labelsB,
                Tensor<std::complex<float>> C, const std::string_view labelsC);

  template <>
  void contract(Tensor<std::complex<double>> A, const std::string_view labelsA,
                Tensor<std::complex<double>> B, const std::string_view labelsB,
                Tensor<std::complex<double>> C, const std::string_view labelsC);

  template <>
  void contract(Tensor<float> A, const std::string_view labelsA,
                Tensor<float> B, const std::string_view labelsB,
                Tensor<float> C, const std::string_view labelsC);

  template <>
  void contract(Tensor<double> A, const std::string_view labelsA,
                Tensor<double> B, const std::string_view labelsB,
                Tensor<double> C, const std::string_view labelsC);
};
