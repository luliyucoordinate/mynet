/**
 * @brief A Ops factory that allows one to register Opss.
 * During runtime, registered Opss can be called by passing a OpsParameterT*
 * protobuffer to the CreateOps function:
 *
 *     OpsRegistry<Dtype>::CreateOps(param);
 *
 * There are two ways to register a Ops. Assuming that we have a Ops like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeOps : public Ops<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Ops" at the end
 * ("MyAwesomeOps" -> "MyAwesome").
 *
 * If the Ops is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_Ops_CLASS(MyAwesome);
 *
 * Or, if the Ops is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Ops<Dtype*> GetMyAwesomeOps(OpsParameterT* param) {
 *      // your implementation
 *    }
 *
 * (for example, when your Ops has multiple backends, see GetConvolutionOps
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_Ops_CREATOR(MyAwesome, GetMyAwesomeOps)
 *
 * Note that each Ops type should only be registered once.
 */

#ifndef MYNET_OPS_FACTORY_H_
#define MYNET_OPS_FACTORY_H_

#include "common.hpp"
#include "ops.hpp"
#include "core/schema/mynet_generated.h"

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
    DCHECK_EQ(registry.count(type), 1ul) << "Unknown Ops type: " << type
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


#define REGISTER_OPS_CREATOR(type, creator)                                  \
  static OpsRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static OpsRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

#define REGISTER_OPS_CLASS(type)                                             \
  template <typename Dtype>                                                  \
  std::shared_ptr<Ops<Dtype>> Creator_##type##Ops(OpsParameterT* param)      \
  {                                                                          \
    return std::make_shared<type##Ops<Dtype>>(param);                      \
  }                                                                          \
  REGISTER_OPS_CREATOR(type, Creator_##type##Ops)

}  // namespace mynet

#endif  // mynet_Ops_FACTORY_H_
