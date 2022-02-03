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

#include "core/schema/filler_generated.h"
#include "core/schema/tensor_generated.h"
#include "core/schema/op_generated.h"
#include "core/schema/mynet_generated.h"
#include "core/schema/solver_generated.h"  // filler, tensor, op, mynet, solver
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
      (temp_files_subpath / format_int(next_temp_file++, 9)).string();
}

bool ReadNetParamsFromTextFile(const char* filename, NetParameterT** flat);

inline bool ReadNetParamsFromTextFile(const std::string& filename,
                                      NetParameterT** flat) {
  return ReadNetParamsFromTextFile(filename.c_str(), flat);
}

void WriteNetParamsToTextFile(const NetParameterT* flat, const char* filename);
inline void WriteNetParamsToTextFile(const NetParameterT* flat,
                                     const std::string& filename) {
  WriteNetParamsToTextFile(flat, filename.c_str());
}

bool ReadNetParamsFromBinaryFile(const char* filename, NetParameterT** flat);

inline bool ReadNetParamsFromBinaryFile(const std::string& filename,
                                        NetParameterT** flat) {
  return ReadNetParamsFromBinaryFile(filename.c_str(), flat);
}

void WriteNetParamsToBinaryFile(const NetParameterT* flat,
                                const char* filename);
inline void WriteNetParamsToBinaryFile(const NetParameterT* flat,
                                       const std::string& filename) {
  WriteNetParamsToBinaryFile(flat, filename.c_str());
}

// Read parameters from a file into a SolverParameter proto message.
void ReadSolverParamsFromTextFile(const std::string& param_file,
                                  SolverParameterT** param);

}  // namespace mynet

#endif  // CORE_LIB_IO_HPP_
