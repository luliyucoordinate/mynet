// Copyright 2021 coordinate
// Author: coordinate

#include <map>
#include <memory>
#include <string>

#include "flatbuffers/flatbuffers.h"
#include "mynet_test_main.hpp"
#include "op_factory.hpp"

namespace mynet {

template <typename TypeParam>
class OpFactoryTest : public MultiDeviceTest<TypeParam> {};

TYPED_TEST_CASE(OpFactoryTest, TestDtypesAndDevices);

TYPED_TEST(OpFactoryTest, TestCreateOp) {
  typedef typename TypeParam::Dtype Dtype;
  typename OpRegistry<Dtype>::CreatorRegistry& registry =
      OpRegistry<Dtype>::Registry();
  std::shared_ptr<Op<Dtype>> op;
  for (const auto& [k, v] : registry) {
    OpParameterT op_param;
    // Data layers expect a DB
    if (k == "Data") {
      continue;
    }

    op_param.type = k;
    op_param.conv_param = std::make_unique<ConvParameterT>();
    op = OpRegistry<Dtype>::CreateOp(&op_param);
    EXPECT_EQ(k, op->type());
  }
}

}  // namespace mynet
