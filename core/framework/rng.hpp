// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_RNG_HPP_
#define CORE_FRAMEWORK_RNG_HPP_

#include <algorithm>
#include <iterator>
#include <random>

#include "common.hpp"

namespace mynet {

typedef std::mt19937 rng_t;

inline rng_t* mynet_rng() {
  return static_cast<mynet::rng_t*>(Mynet::rng_stream().generator());
}

// Fisher–Yates algorithm
template <class RandomAccessIterator, class RandomGenerator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end,
                    RandomGenerator* gen) {
  typedef typename std::iterator_traits<RandomAccessIterator>::difference_type
      difference_type;
  typedef typename std::uniform_real_distribution<difference_type> dist_type;

  difference_type length = std::distance(begin, end);
  if (length <= 0) return;

  for (difference_type i = length - 1; i > 0; --i) {
    dist_type dist(0, i);
    std::iter_swap(begin + i, begin + dist(*gen));
  }
}

template <class RandomAccessIterator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end) {
  shuffle(begin, end, mynet_rng());
}
}  // namespace mynet

#endif  // CORE_FRAMEWORK_RNG_HPP_
