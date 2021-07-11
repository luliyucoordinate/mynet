#include "common.hpp"

namespace mynet {

void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}

thread_local static std::unique_ptr<Mynet> thread_instance_;

Mynet& Mynet::Get() {
  if (!thread_instance_.get()) {
    thread_instance_.reset(new Mynet());
  }
  return *(thread_instance_.get());
}

Mynet::Mynet()
    : mode_(Mynet::CPU) { }

Mynet::~Mynet() { }

}  // namespace mynet