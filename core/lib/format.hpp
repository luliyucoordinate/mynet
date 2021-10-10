// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_LIB_FORMAT_HPP_
#define CORE_LIB_FORMAT_HPP_

#include <iomanip>
#include <sstream>
#include <string>

namespace mynet {

inline std::string format_int(int n, int zeros = 0) {
  std::ostringstream s;
  s << std::setw(zeros) << std::setfill('0') << n;
  return s.str();
}

}  // namespace mynet

#endif  // CORE_LIB_FORMAT_HPP_
