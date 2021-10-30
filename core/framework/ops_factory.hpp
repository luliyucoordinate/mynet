// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_OPS_FACTORY_HPP_
#define CORE_FRAMEWORK_OPS_FACTORY_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "ops.hpp"

namespace mynet {

template <typename Dtype>
class Ops;

template <typename Dtype>
class OpsRegistry {
 public:
  typedef std::shared_ptr<Ops<Dtype>> (*Creator)(OpsParameterT*);
  typedef std::map<std::string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const std::string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    DCHECK_EQ(registry.count(type), 0ul)
        << "Ops type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a Ops using a OpsParameterT*.
  static std::shared_ptr<Ops<Dtype>> CreateOps(OpsParameterT* param) {
    if (Mynet::root_solver()) {
      LOG(INFO) << "Creating Ops " << param->name;
    }
    auto type = param->type;
    CreatorRegistry& registry = Registry();
    DCHECK_EQ(registry.count(type), 1ul)
        << "Unknown Ops type: " << type
        << " (known types: " << OpsTypeListString() << ")";
    return registry[type](param);
  }

  static std::vector<std::string> OpsTypeList() {
    CreatorRegistry& registry = Registry();
    std::vector<std::string> Ops_types;
    for (const auto& [k, v] : registry) {
      Ops_types.push_back(k);
    }
    return Ops_types;
  }

 private:
  // Ops registry should never be instantiated - everything is done with its
  // static variables.
  OpsRegistry() {}

  static std::string OpsTypeListString() {
    std::vector<std::string> Ops_types = OpsTypeList();
    std::string Ops_types_str;
    for (auto iter = Ops_types.begin(); iter != Ops_types.end(); ++iter) {
      if (iter != Ops_types.begin()) {
        Ops_types_str += ", ";
      }
      Ops_types_str += *iter;
    }
    return Ops_types_str;
  }
};

template <typename Dtype>
class OpsRegisterer {
 public:
  OpsRegisterer(const std::string& type,
                std::shared_ptr<Ops<Dtype>> (*creator)(OpsParameterT*)) {
    OpsRegistry<Dtype>::AddCreator(type, creator);
  }
};

#define REGISTER_OPS_CREATOR(type, creator)                              \
  static OpsRegisterer<float> g_creator_f_##type(#type, creator<float>); \
  static OpsRegisterer<double> g_creator_d_##type(#type, creator<double>)

#define REGISTER_OPS_CLASS(type)                                          \
  template <typename Dtype>                                               \
  std::shared_ptr<Ops<Dtype>> Creator_##type##Ops(OpsParameterT* param) { \
    return std::make_shared<type##Ops<Dtype>>(param);                     \
  }                                                                       \
  REGISTER_OPS_CREATOR(type, Creator_##type##Ops)

}  // namespace mynet

#endif  // CORE_FRAMEWORK_OPS_FACTORY_HPP_
