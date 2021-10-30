// Copyright 2021 coordinate
// Author: coordinate

#include "io.hpp"

#include <fcntl.h>
#include <flatbuffers/idl.h>
#include <flatbuffers/util.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace mynet {

bool ReadFlatFromTextFile(const char* filename, TensorFlatT** tensor_flat) {
  std::string schema_file, json_file;
  bool success =
      flatbuffers::LoadFile("core/schema/tensor.fbs", false, &schema_file) &&
      flatbuffers::LoadFile(filename, false, &json_file);
  DCHECK(success) << "File not found: " << filename;

  flatbuffers::Parser parser;
  success =
      parser.Parse(schema_file.c_str()) && parser.Parse(json_file.c_str());
  std::cout << parser.error_ << std::endl;
  DCHECK(success) << "Parse file error: " << filename;
  *tensor_flat =
      flatbuffers::GetRoot<TensorFlat>(parser.builder_.GetBufferPointer())
          ->UnPack();
  return true;
}

void WriteFlatToTextFile(const TensorFlatT* tensor_flat, const char* filename) {
  std::ofstream output_file(filename);
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(TensorFlat::Pack(fbb, tensor_flat));

  std::string schema_file;
  DCHECK(flatbuffers::LoadFile("core/schema/tensor.fbs", false, &schema_file));

  flatbuffers::Parser parser;
  DCHECK(parser.Parse(schema_file.c_str()));

  std::string jsongen;
  DCHECK(flatbuffers::GenerateText(
      parser, reinterpret_cast<const void*>(fbb.GetBufferPointer()), &jsongen))
      << "Flat to File error: " << filename;
  output_file << jsongen;
}

bool ReadFlatFromBinaryFile(const char* filename, TensorFlatT** tensor_flat) {
  std::string data;
  DCHECK(flatbuffers::LoadFile(filename, true, &data));
  *tensor_flat = const_cast<TensorFlatT*>(
      flatbuffers::GetRoot<TensorFlat>(data.c_str())->UnPack());
  return true;
}

void WriteFlatToBinaryFile(const TensorFlatT* tensor_flat,
                           const char* filename) {
  DCHECK(tensor_flat) << "write empty file";
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(TensorFlat::Pack(fbb, tensor_flat));
  DCHECK(flatbuffers::SaveFile(filename,
                               reinterpret_cast<char*>(fbb.GetBufferPointer()),
                               fbb.GetSize(), true));
}

}  // namespace mynet
