// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_LIB_IO_HPP_
#define CORE_LIB_IO_HPP_

#include <glog/logging.h>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "core/schema/tensor_generated.h"
#include "format.hpp"

#ifndef MYNET_TMP_DIR_RETRIES
#define MYNET_TMP_DIR_RETRIES 100
#endif

namespace mynet {

inline void MakeTempDir(std::string* temp_dirname) {
  temp_dirname->clear();
  auto model = std::filesystem::temp_directory_path() / "mynet_test.%%%%-%%%%";
  for (int i = 0; i < MYNET_TMP_DIR_RETRIES; i++) {
    bool done = std::filesystem::create_directory(model);
    if (done) {
      *temp_dirname = model.string();
      return;
    }
  }
  LOG(FATAL) << "Failed to create a temporary directory.";
}

inline void MakeTempFilename(std::string* temp_filename) {
  static std::filesystem::path temp_files_subpath;
  static uint64_t next_temp_file = 0;
  temp_filename->clear();
  if (temp_files_subpath.empty()) {
    std::string path_string = "";
    MakeTempDir(&path_string);
    temp_files_subpath = path_string;
  }
  *temp_filename =
      (temp_files_subpath / mynet::format_int(next_temp_file++, 9)).string();
}

bool ReadFlatFromTextFile(const char* filename, TensorFlatT** tensor_flat);

inline bool ReadFlatFromTextFile(const std::string& filename,
                                 TensorFlatT** flat) {
  return ReadFlatFromTextFile(filename.c_str(), flat);
}

// inline void ReadFlatFromTextFileOrDie(const char* filename, TensorFlatT**
// flat) {
//   CHECK(ReadFlatFromTextFile(filename, flat));
// }

// inline void ReadFlatFromTextFileOrDie(const std::string& filename,
// TensorFlatT** flat) {
//   ReadFlatFromTextFileOrDie(filename.c_str(), flat);
// }

void WriteFlatToTextFile(const TensorFlatT* flat, const char* filename);
inline void WriteFlatToTextFile(const TensorFlatT* flat,
                                const std::string& filename) {
  WriteFlatToTextFile(flat, filename.c_str());
}

bool ReadFlatFromBinaryFile(const char* filename, TensorFlatT** flat);

inline bool ReadFlatFromBinaryFile(const std::string& filename,
                                   TensorFlatT** flat) {
  return ReadFlatFromBinaryFile(filename.c_str(), flat);
}

// inline void ReadFlatFromBinaryFileOrDie(const char* filename, TensorFlatT**
// flat) {
//   CHECK(ReadFlatFromBinaryFile(filename, flat));
// }

// inline void ReadFlatFromBinaryFileOrDie(const std::string& filename,
// TensorFlatT** flat) {
//   ReadFlatFromBinaryFileOrDie(filename.c_str(), flat);
// }

void WriteFlatToBinaryFile(const TensorFlatT* flat, const char* filename);
inline void WriteFlatToBinaryFile(const TensorFlatT* flat,
                                  const std::string& filename) {
  WriteFlatToBinaryFile(flat, filename.c_str());
}

}  // namespace mynet

#endif  // CORE_LIB_IO_HPP_
