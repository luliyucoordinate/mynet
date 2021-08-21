#include "ops.hpp"
#include "ops_factory.hpp"
#include "core/protobuf/mynet.pb.h"

namespace mynet {

// Get convolution ops according to engine.
template <typename Dtype>
std::shared_ptr<Ops<Dtype>> GetConvolutionOps(
    const OpsParameter& param) {
  ConvolutionParameter conv_param = param.convolution_param();
  ConvolutionParameter_Engine engine = conv_param.engine();
  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_MYNET;
  }
  if (engine == ConvolutionParameter_Engine_MYNET) {
    return std::shared_ptr<Ops<Dtype> >(new ConvolutionOps<Dtype>(param));
  } else {
    LOG(FATAL) << "ops " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_OPS_CREATOR(Convolution, GetConvolutionOps);

// Ops that use their constructor as their default creator should be
// registered in their corresponding cpp files. Do not register them here.
}  // namespace mynet
