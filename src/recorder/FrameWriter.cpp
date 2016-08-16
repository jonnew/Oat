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

#include <iostream>
#include <cassert>

namespace oat {

void FrameWriter::initialize(const std::string &source_name,
                             const oat::Frame &f) {

    // Initialize writer using the first frame taken from server
    int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    video_writer_.open(path_, fourcc, f.sample().rate_hz(), f.size());
}

void FrameWriter::write(void) {

    cv::Mat mat;
    while (buffer_.pop(mat)) {

        // File desriptor must be avaiable for writing
        assert(video_writer_.isOpened());

        video_writer_.write(mat);
    }
}

} /* namespace oat */
