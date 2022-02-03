// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_LIB_FORMAT_HPP_
#define CORE_LIB_FORMAT_HPP_

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace mynet {

inline std::string format_int(uint64_t n, uint32_t zeros = 0) {
  std::ostringstream s;
  s << std::setw(zeros) << std::setfill('0') << n;
  return s.str();
}

std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

// trim from start (in place)
void ltrim(std::string *s) {
  s->erase(s->begin(), std::find_if(s->begin(), s->end(),
                                    [](char ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
void rtrim(std::string *s) {
  s->erase(std::find_if(s->rbegin(), s->rend(),
                        [](char ch) { return !std::isspace(ch); })
               .base(),
           s->end());
}

// trim from both ends (in place)
void trim(std::string *s) {
  ltrim(s);
  rtrim(s);
}

}  // namespace mynet

#endif  // CORE_LIB_FORMAT_HPP_
