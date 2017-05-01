//******************************************************************************
//* File:   FrameWriter.cpp
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

#include "FrameWriter.h"

#include <cassert>

#include "../../lib/utility/FileFormat.h"
#include "../../lib/utility/IOFormat.h"

namespace oat {

void FrameWriter::configure(const oat::config::OptionTable &t,
                            const po::variables_map &vm)
{
    // File overwrite
    oat::config::getValue(vm, t, "allow-overwrite", allow_overwrite_);

    // Compression level
    std::string fcc;
    if (oat::config::getValue(vm, t, "fourcc", fcc)) {

        if (fcc.size() != 4)
            throw std::runtime_error("fourcc must be 4 characters long.");

        if (fcc == "none" || fcc == "None" || fcc == "NONE")
            fourcc_ = 0;
        else
            fourcc_ = cv::VideoWriter::fourcc(fcc[0], fcc[1], fcc[2], fcc[3]);

        if (fourcc_ < 0)
            throw std::runtime_error("Unsupported fourcc code.");
    }
}

void FrameWriter::close()
{
    video_writer_.release();
}

void FrameWriter::initialize(const std::string &path)
{
    path_ = path + ".avi";

    if (!allow_overwrite_)
       oat::ensureUniquePath(path_);

    if (!oat::checkWritePermission(path_))
        throw std::runtime_error("Write permission denied for " + path_);

    auto frame = source_.retrieve();
    auto sz = cv::Size(frame->cols(), frame->rows());

    if (!video_writer_.open(path_,
                            fourcc_,
                            1.0 / frame->period<Token::Seconds>().count(),
                            sz,
                            Pixel::multichomatic(frame->color()))) {
        throw std::runtime_error("Could open video writer.");
    }
}

void FrameWriter::write(void)
{
    buffer_.consume_all(
        [this](oat::Frame &f) { video_writer_.write(f.mat()); });
}

int FrameWriter::pullToken(void )
{
    // Synchronous pull from source
    oat::Frame frame;
    auto rc = source_.pull(frame);
    if (rc) { return rc; }

    // Move frame to buffer
    if (!buffer_.push(std::move(frame))) 
        throw std::runtime_error(overrun_msg);

    return rc; 
}

} /* namespace oat */
