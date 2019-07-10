/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

template <typename T>
class GeluOp : public tensorflow::OpKernel {
 public:
  explicit GeluOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    // TODO: handle any data type for input and output
    auto output_flat = output_tensor->flat<T>();

    // Set the value of each element
    const int N = input.size();
    auto sqrt2overPI = T(sqrt(2/M_PI));
    // TODO: investigate if we can optimize using other TF ops or using MKLDNN or Eigen
    for (int i = 0; i < N; i++) {
      auto x = input(i);

      auto cdf = T(0.5) * (T(1.0) + tanh(
        sqrt2overPI * (x + T(0.044715)*pow(x, T(3)))));
      auto y = x * cdf;

      output_flat(i) = y;
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Gelu")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<Eigen::half>("T"),
    GeluOp<Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("Gelu")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<float>("T"),
    GeluOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Gelu")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<double>("T"),
    GeluOp<double>);


template <typename T>
class GeluGradOp : public tensorflow::OpKernel {
 public:
  explicit GeluGradOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // Grab the grad input tensor
    const Tensor& grad_input_tensor = context->input(0);
    auto grad_input = grad_input_tensor.flat<T>();

    // Grab the input tensor
    const Tensor& input_tensor = context->input(1);
    auto input = input_tensor.flat<T>();

    // Grab the gelu output tensor
    const Tensor& output_tensor = context->input(2);
    auto output = output_tensor.flat<T>();

    // Create an output tensor
    Tensor* grad_output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_input_tensor.shape(),
                                                     &grad_output_tensor));
    // TODO: handle any data type for input and output
    auto grad_output_flat = grad_output_tensor->flat<T>();

    // Set the value of each element
    const int N = input.size();
    auto sqrt2overPI = T(sqrt(2/M_PI));
    // TODO: investigate if we can optimize using other TF ops or using MKLDNN or Eigen
    for (int i = 0; i < N; i++) {
      auto dL_dy = grad_input(i);
      auto x = input(i);
      auto gelu_x = output(i) / x;

      auto tanhterm = tanh(sqrt2overPI * (x + T(0.04715)*pow(x,T(3))));
      auto dy_dx = gelu_x + T(0.5) * x * (T(1) - pow(tanhterm,T(2))) * sqrt2overPI * (T(1) + T(3) * T(0.04715) * pow(x,T(2)));
      auto dL_dx = dL_dy * dy_dx;

      grad_output_flat(i) = dL_dx;
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("GeluGrad")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<Eigen::half>("T"),
    GeluGradOp<Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("GeluGrad")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<float>("T"),
    GeluGradOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("GeluGrad")
    .Device(tensorflow::DEVICE_CPU)
    .TypeConstraint<double>("T"),
    GeluGradOp<double>);


}  // namespace tensorflow


