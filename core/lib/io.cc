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

bool ReadNetParamsFromTextFile(const char* filename, NetParameterT** flat) {
  std::string schema_file, json_file;
  DCHECK(flatbuffers::LoadFile("core/schema/mynet.fbs", false, &schema_file))
      << "Load schema file error";
  DCHECK(flatbuffers::LoadFile(filename, false, &json_file))
      << "File not found: " << filename;

  flatbuffers::Parser parser;
  const char* include_directories[] = {"core/schema", nullptr};
  DCHECK(parser.Parse(schema_file.c_str(), include_directories))
      << parser.error_;
  DCHECK(parser.Parse(json_file.c_str())) << parser.error_;
  *flat = flatbuffers::GetRoot<NetParameter>(parser.builder_.GetBufferPointer())
              ->UnPack();
  return true;
}

void WriteNetParamsToTextFile(const NetParameterT* flat, const char* filename) {
  std::ofstream output_file(filename);
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(NetParameter::Pack(fbb, flat));

  std::string schema_file;
  DCHECK(flatbuffers::LoadFile("core/schema/mynet.fbs", false, &schema_file));

  flatbuffers::Parser parser;
  const char* include_directories[] = {"core/schema", nullptr};
  DCHECK(parser.Parse(schema_file.c_str(), include_directories))
      << parser.error_;

  std::string jsongen;
  DCHECK(flatbuffers::GenerateText(
      parser, reinterpret_cast<const void*>(fbb.GetBufferPointer()), &jsongen))
      << "NetParams to File error: " << filename;
  output_file << jsongen;
}

bool ReadNetParamsFromBinaryFile(const char* filename, NetParameterT** flat) {
  std::string data;
  DCHECK(flatbuffers::LoadFile(filename, true, &data));
  *flat = flatbuffers::GetRoot<NetParameter>(data.c_str())->UnPack();
  return true;
}

void WriteNetParamsToBinaryFile(const NetParameterT* flat,
                                const char* filename) {
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(NetParameter::Pack(fbb, flat));
  DCHECK(flatbuffers::SaveFile(filename,
                               reinterpret_cast<char*>(fbb.GetBufferPointer()),
                               fbb.GetSize(), true));
}

}  // namespace mynet
