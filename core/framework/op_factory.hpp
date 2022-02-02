// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_OP_FACTORY_HPP_
#define CORE_FRAMEWORK_OP_FACTORY_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "op.hpp"

namespace mynet {

template <typename Dtype>
class Op;

template <typename Dtype>
class OpRegistry {
 public:
  typedef std::shared_ptr<Op<Dtype>> (*Creator)(OpParameterT*);
  typedef std::map<std::string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const std::string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    DCHECK_EQ(registry.count(type), 0ul)
        << "Op type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a Op using a OpParameterT*.
  static std::shared_ptr<Op<Dtype>> CreateOp(OpParameterT* param) {
    if (Mynet::root_solver()) {
      LOG(INFO) << "Creating Op " << param->name;
    }
    const auto& type = param->type;
    CreatorRegistry& registry = Registry();
    DCHECK_EQ(registry.count(type), 1ul)
        << "Unknown Op type: " << type
        << " (known types: " << OpTypeListString() << ")";
    return registry[type](param);
  }

  static std::vector<std::string> OpTypeList() {
    CreatorRegistry& registry = Registry();
    std::vector<std::string> op_types;
    for (const auto& [k, v] : registry) {
      op_types.push_back(k);
    }
    return op_types;
  }

 private:
  // Op registry should never be instantiated - everything is done with its
  // static variables.
  OpRegistry() {}

  static std::string OpTypeListString() {
    std::vector<std::string> op_types = OpTypeList();
    std::string op_types_str;
    for (auto iter = op_types.begin(); iter != op_types.end(); ++iter) {
      if (iter != op_types.begin()) {
        op_types_str += ", ";
      }
      op_types_str += *iter;
    }
    return op_types_str;
  }
};

template <typename Dtype>
class OpRegisterer {
 public:
  OpRegisterer(const std::string& type,
               std::shared_ptr<Op<Dtype>> (*creator)(OpParameterT*)) {
    OpRegistry<Dtype>::AddCreator(type, creator);
  }
};

#define REGISTER_OP_CREATOR(type, creator)                              \
  static OpRegisterer<float> g_creator_f_##type(#type, creator<float>); \
  static OpRegisterer<double> g_creator_d_##type(#type, creator<double>)

#define REGISTER_OP_CLASS(type)                                        \
  template <typename Dtype>                                            \
  std::shared_ptr<Op<Dtype>> Creator_##type##Op(OpParameterT* param) { \
    return std::make_shared<type##Op<Dtype>>(param);                   \
  }                                                                    \
  REGISTER_OP_CREATOR(type, Creator_##type##Op)

}  // namespace mynet

#endif  // CORE_FRAMEWORK_OP_FACTORY_HPP_
