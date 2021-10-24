// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_COMMON_HPP_
#define CORE_FRAMEWORK_COMMON_HPP_

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

#define DISABLE_COPY_AND_ASSIGN(classname) \
 private:                                  \
  classname(const classname&);             \
  classname& operator=(const classname&)

#endif  // CORE_FRAMEWORK_COMMON_HPP_
