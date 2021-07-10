// #include "common.hpp"
// #include "filler.hpp"
// #include "mynet_test_main.hpp"

// namespace mynet {

// template <typename Dtype>
// class UniformFillerTest : public ::testing::Test {
//  protected:
//   UniformFillerTest()
//       : tensor_(new Tensor<Dtype>()),
//         filler_param_() {
//     filler_param_.set_min(1.);
//     filler_param_.set_max(2.);
//     filler_.reset(new UniformFiller<Dtype>(filler_param_));
//   }
//   virtual void test_params(const std::vector<int>& shape) {
//     EXPECT_TRUE(tensor_);
//     tensor_->Reshape(shape);
//     filler_->Fill(tensor_);
//     const int count = tensor_->count();
//     const Dtype* data = tensor_->cpu_data();
//     for (int i = 0; i < count; ++i) {
//       EXPECT_GE(data[i], filler_param_.min());
//       EXPECT_LE(data[i], filler_param_.max());
//     }
//   }
//   virtual ~UniformFillerTest() { delete tensor_; }
//   Tensor<Dtype>* const tensor_;
//   FillerParameter filler_param_;
//   std::shared_ptr<UniformFiller<Dtype>> filler_;
// };

// TYPED_TEST_CASE(UniformFillerTest, TestDtypes);

// TYPED_TEST(UniformFillerTest, TestFill) {
//   std::vector<int> tensor_shape;
//   tensor_shape.push_back(2);
//   tensor_shape.push_back(3);
//   tensor_shape.push_back(4);
//   tensor_shape.push_back(5);
//   this->test_params(tensor_shape);
// }

// TYPED_TEST(UniformFillerTest, TestFill1D) {
//   std::vector<int> tensor_shape(1, 15);
//   this->test_params(tensor_shape);
// }

// TYPED_TEST(UniformFillerTest, TestFill2D) {
//   std::vector<int> tensor_shape;
//   tensor_shape.push_back(8);
//   tensor_shape.push_back(3);
//   this->test_params(tensor_shape);
// }

// TYPED_TEST(UniformFillerTest, TestFill5D) {
//   std::vector<int> tensor_shape;
//   tensor_shape.push_back(2);
//   tensor_shape.push_back(3);
//   tensor_shape.push_back(4);
//   tensor_shape.push_back(5);
//   tensor_shape.push_back(2);
//   this->test_params(tensor_shape);
// }

// }  // namespace mynet