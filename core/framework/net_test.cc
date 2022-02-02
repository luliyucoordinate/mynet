// Copyright 2021 coordinate
// Author: coordinate

#include "net.hpp"

#include <string>
#include <utility>
#include <vector>

#include "core/lib/io.hpp"
#include "mynet_test_main.hpp"

namespace mynet {

template <typename TypeParam>
class NetTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  NetTest() : seed_(1701) {}

  virtual void InitNetFromTextFile(const char* filename) {
    auto param_t = std::make_shared<NetParameterT>();
    auto param = param_t.get();
    ReadNetParamsFromTextFile(filename, &param);
    net_.reset(new Net<Dtype>(param));
  }

  virtual void InitNetFromTextFileWithState(
      const char* filename, Phase phase = mynet::Phase_TRAIN,
      uint32_t level = 0, const std::vector<std::string>* stages = nullptr) {
    auto param_t = std::make_shared<NetParameterT>();
    auto param = param_t.get();
    ReadNetParamsFromTextFile(filename, &param);
    std::string param_file;
    MakeTempFilename(&param_file);
    WriteNetParamsToTextFile(param, param_file.c_str());
    net_.reset(new Net<Dtype>(param_file, phase, level, stages));
  }

  virtual void CopyNetTensors(
      bool copy_diff,
      std::vector<std::shared_ptr<Tensor<Dtype>>>* tensors_copy) {
    DCHECK(net_);
    auto net_tensors = net_->tensors();
    tensors_copy->clear();
    tensors_copy->resize(net_tensors.size());
    bool kReshape = true;
    for (uint32_t i = 0; i < net_tensors.size(); ++i) {
      (*tensors_copy)[i].reset(new Tensor<Dtype>());
      (*tensors_copy)[i]->CopyFrom(*net_tensors[i], copy_diff, kReshape);
    }
  }

  virtual void CopyNetParams(
      bool copy_diff,
      std::vector<std::shared_ptr<Tensor<Dtype>>>* params_copy) {
    DCHECK(net_);
    auto net_params = net_->params();
    params_copy->clear();
    params_copy->resize(net_params.size());
    bool kReshape = true;
    for (uint32_t i = 0; i < net_params.size(); ++i) {
      (*params_copy)[i].reset(new Tensor<Dtype>());
      (*params_copy)[i]->CopyFrom(*net_params[i], copy_diff, kReshape);
    }
  }

  virtual void InitTinyNetEuclidean() {
    const char* filename_r = "core/test_data/tinynet.json";
    InitNetFromTextFile(filename_r);
  }

  uint32_t seed_;
  std::shared_ptr<Net<Dtype>> net_;
};

TYPED_TEST_CASE(NetTest, TestDtypesAndDevices);

TYPED_TEST(NetTest, TestBottomNeedBackwardEuclideanForce) {
  this->InitTinyNetEuclidean();
  auto input_need_backward = this->net_->input_need_backward();
  EXPECT_EQ(3ul, input_need_backward.size());
  EXPECT_EQ(0ul, input_need_backward[0].size());
  EXPECT_EQ(1ul, input_need_backward[1].size());
  EXPECT_EQ(true, input_need_backward[1][0]);
  EXPECT_EQ(2ul, input_need_backward[2].size());
  EXPECT_EQ(true, input_need_backward[2][0]);
  EXPECT_EQ(true, input_need_backward[2][1]);
}

}  // namespace mynet
