// Copyright 2021 coordinate
// Author: coordinate

#include "common.hpp"

#include <memory>

#include "rng.hpp"

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
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO) << "System entropy source not available, "
               "using fallback algorithm to generate seed instead.";
  if (f) fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
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
  Generator() : rng_(new mynet::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new mynet::rng_t(seed)) {}
  mynet::rng_t* rng() { return rng_.get(); }

 private:
  std::shared_ptr<mynet::rng_t> rng_;
};

Mynet::RNG::RNG() : generator_(new Generator()) {}

Mynet::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) {}

Mynet::RNG& Mynet::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Mynet::RNG::generator() { return static_cast<void*>(generator_->rng()); }

}  // namespace mynet
