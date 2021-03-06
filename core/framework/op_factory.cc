// Copyright 2021 coordinate
// Author: coordinate

#include "op_factory.hpp"

#include <memory>

#include "core/kernels/conv_ops.hpp"
#include "core/kernels/softmax_op.hpp"
#include "op.hpp"

namespace mynet {

// Get conv op according to engine.
template <typename Dtype>
std::shared_ptr<Op<Dtype>> GetConvOp(OpParameterT* param) {
  DCHECK(param);
  auto& conv_param = param->conv_param;
  DCHECK(conv_param);
  auto& engine = conv_param->engine;
  if (engine == Engine_DEFAULT) {
    engine = Engine_MYNET;
  }
  if (engine == Engine_MYNET) {
    return std::make_shared<ConvOp<Dtype>>(param);
  } else {
    LOG(FATAL) << "op " << param->name << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_OP_CREATOR(Conv, GetConvOp);

// Get softmax layer according to engine.
template <typename Dtype>
std::shared_ptr<Op<Dtype>> GetSoftmaxOp(OpParameterT* param) {
  DCHECK(param);
  auto& softmax_param = param->softmax_param;
  DCHECK(softmax_param);
  auto& engine = softmax_param->engine;
  if (engine == Engine_DEFAULT) {
    engine = Engine_MYNET;
  }
  if (engine == Engine_MYNET) {
    return std::make_shared<SoftmaxOp<Dtype>>(param);
  } else {
    LOG(FATAL) << "Op " << param->name << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_OP_CREATOR(Softmax, GetSoftmaxOp);

// Op that use their constructor as their default creator should be
// registered in their corresponding cpp files. Do not register them here.
}  // namespace mynet
