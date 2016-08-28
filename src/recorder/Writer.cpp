//******************************************************************************
//* File:   Writer.cpp
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

#include "Writer.h"

Writer<oat::Position2D>::~Writer
{
    json_writer_.EndArray();
    json_writer_.EndObject();
    file_stream_->Flush();
}

void Writer<oat::Position2D>::initialize(
        const std::string &source_name,
        const Position2D &sample_template) {

    // Position file
    fd_ = fopen(path_.c_str(), "wb");

    file_stream_.reset(new rapidjson::FileWriteStream(fd_,
            position_write_buffer, sizeof(position_write_buffer)));
    json_writer_.Reset(*file_stream_);

    // Main object, end this object before write flush in destructor
    json_writer_.StartObject();

    // Version
    json_writer_.StartObject();

    // Oat version in header 
    char version[255];
    strcpy (version, Oat_VERSION_MAJOR);
    strcat (version, ".");
    strcat (version, Oat_VERSION_MINOR);
    json_writer_.String("oat_version");
    json_writer_.String(version);

    // Complete header object
    json_writer_.String("header");

    json_writer_.String("date");
    json_writer_.String("TODO");
    //json_writer_.String(date.c_str());

    auto fs = p.sample().rate_hz();
    json_writer_.String("sample_rate_hz");
    if (std::isfinite(fs))
        json_writer_.Double(fs);
    else
        json_writer_.Double(-1.0);

    json_writer_.String("source");
    json_writer_.String(source_name.c_str());

    json_writer_.EndObject();

    // Start data object
    json_writer_.String("positions");
    json_writer_.StartArray();
}

void Writer<oat::Position2D>::write(void) {

   oat::Position2D p("");
   while (buffer_.pop(p)) {

       // File desriptor must be avaiable for writing
       assert(fd_);

       json_writer_.StartObject();
       json_writer_.String("TODO: timestamp");
       //json_writer_.String(p.label());
       p.Serialize(json_writer_, verbose_file_);

       json_writer_.EndObject();
   }
}

void Writer<oat::Frame>initialize(
        const std::string &source_name,
        const Position2D &sample_template) {

   // Initialize writer using the first frame taken from server
   int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
   video_writer_.open(path_, fourcc, f.sample().rate_hz(), f.size());
}

void Writer<oat::Frame>::write(void) {

   cv::Mat mat;
   while (buffer_.pop(mat)) {

       assert(video_writer_.isOpened());
       video_writer_.write(mat);
   }
}
