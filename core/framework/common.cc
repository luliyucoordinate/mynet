// Copyright 2021 coordinate
// Author: coordinate

#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <memory>
#include <random>

namespace mynet {

thread_local static std::unique_ptr<Mynet> thread_instance_;

Mynet& Mynet::Get() {
  if (!thread_instance_.get()) {
    thread_instance_.reset(new Mynet());
  }
  return *(thread_instance_.get());
}

// random seeding
int64_t cluster_seedgen(void) {
  return std::chrono::system_clock::now().time_since_epoch().count();
}

void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}

Mynet::Mynet() : mode_(Mynet::CPU) {}

Mynet::~Mynet() {}

void Mynet::set_random_seed(uint32_t seed) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

class Mynet::RNG::Generator {
 public:
  Generator() : rng_(new std::mt19937(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new std::mt19937(seed)) {}
  std::mt19937* rng() { return rng_.get(); }

 private:
  std::shared_ptr<std::mt19937> rng_;
};

Mynet::RNG::RNG() : generator_(new Generator()) {}

Mynet::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) {}

Mynet::RNG& Mynet::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Mynet::RNG::generator() { return static_cast<void*>(generator_->rng()); }

}  // namespace mynet
