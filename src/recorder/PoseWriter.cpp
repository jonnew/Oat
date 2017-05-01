//******************************************************************************
//* File:   PoseWriter.cpp
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu)
//* All right reserved.
//* This file is part of the Oat project.
//* This is free software: you can redistribute it and/or modify
//* it under the terms of the GNU General Public License as published by
//* the Free Software Foundation, either version 3 of the License, or
//* (at your option) any later version.
//* This software is distributed in the hope that it will be useful,
//* but WITHOUT ANY WARRANTY; without even the implied warranty of
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//* GNU General Public License for more details.
//* You should have received a copy of the GNU General Public License
//* along with this source code.  If not, see <http://www.gnu.org/licenses/>.
//*****************************************************************************

#include "PoseWriter.h"
#include "Format.h"

#include <cassert>

#include "../../lib/utility/FileFormat.h"

namespace oat {

PoseWriter::~PoseWriter()
{
    close();
}

void PoseWriter::configure(const oat::config::OptionTable &t,
                               const po::variables_map &vm)
{
    // File overwrite
    oat::config::getValue(vm, t, "allow-overwrite", allow_overwrite_);

    // Write to binary file instead of JSON
    oat::config::getValue(vm, t, "binary-file", use_binary_);

    // Consise JSON file
    oat::config::getValue(vm, t, "concise-file", concise_file_);
}

void PoseWriter::close() 
{
    if (use_binary_ && fd_ != nullptr) {
        auto n = std::to_string(completed_writes_);
        emplaceNumpyShape(fd_, completed_writes_);
        fclose(fd_);
    } else if (fd_ != nullptr) {
        json_writer_.EndArray();
        json_writer_.EndObject();
        file_stream_->Flush();
    }
}

void PoseWriter::initialize(const std::string &path)
{
    if (use_binary_)
        initializeBinary(path);
    else
        initializeJSON(path);
}

void PoseWriter::initializeBinary(const std::string &path)
{
    auto p = path + ".npy";

    if (!allow_overwrite_)
       oat::ensureUniquePath(p);

    if (!oat::checkWritePermission(p))
        throw std::runtime_error("Write permission denied for " + p);

    // Position file
    fd_ = fopen(p.c_str(), "wb");

    // File descriptor must be available for writing
    assert(fd_);

    // Write header
    auto header = getNumpyHeader(oat::POSE_NPY_DTYPE);
    fwrite(header.data(), 1, header.size(), fd_);
}

void PoseWriter::initializeJSON(const std::string &path)
{
    auto p = path + ".json";

    if (!allow_overwrite_)
       oat::ensureUniquePath(p);

    if (!oat::checkWritePermission(p))
        throw std::runtime_error("Write permission denied for " + p);

    // Position file
    fd_ = fopen(p.c_str(), "wb");

    // File descriptor must be available for writing
    assert(fd_);

    file_stream_.reset(new rapidjson::FileWriteStream(
            fd_,
            pose_write_buffer_,
            sizeof(pose_write_buffer_)));
    json_writer_.Reset(*file_stream_);

    // Main object, end this object before write flush in destructor
    json_writer_.StartObject();

    // Oat version
    char version[255];
    strcpy(version, Oat_VERSION_MAJOR);
    strcat(version, ".");
    strcat(version, Oat_VERSION_MINOR);
    json_writer_.String("oat_version");
    json_writer_.String(version);

    // Header object
    json_writer_.String("header");

    json_writer_.StartObject();
    json_writer_.String("date");
    json_writer_.String(oat::createTimeStamp(true).c_str());

    double fs = 1 / sample_period_sec();
    json_writer_.String("sample_rate_hz");
    if (std::isfinite(fs))
        json_writer_.Double(fs);
    else
        json_writer_.Double(-1.0);

    // End header
    json_writer_.EndObject();

    // Start data object
    json_writer_.String("poses");
    json_writer_.StartArray();
}

void PoseWriter::write() {

    buffer_.consume_all([this](oat::Pose &p) { 
        if (use_binary_) {
            auto pack = oat::packPose(p);
            fwrite(pack.data(), 1, pack.size(), fd_);
        } else {
            oat::serializePose(p, json_writer_, !concise_file_);
        }
        completed_writes_++;
    });
}

int PoseWriter::pullToken()
{
    // Synchronous pull from source
    std::unique_ptr<oat::Pose> pose;
    auto rc = source_.pull(pose);
    if (rc) { return rc; }
    
    // Move pose to buffer
    if (!buffer_.push(std::move(*pose)))
        throw std::runtime_error(overrun_msg);

    return rc;
}

} /* namespace oat */
