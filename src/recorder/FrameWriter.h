//******************************************************************************
//* File:   FrameWriter.h
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

#ifndef OAT_FRAMEWRITER_H
#define OAT_FRAMEWRITER_H

#include "Writer.h"

#include <opencv2/videoio.hpp>

#include "../../lib/datatypes/Frame.h"

namespace oat {
namespace blf = boost::lockfree;

// Constants
static constexpr int FRAME_WRITE_BUFFER_SIZE {1000};

/**
 * Frame stream video file writer.
 */
class FrameWriter : public Writer<oat::Frame> {

    // Inherit constructor
    using Writer<oat::Frame>::Writer;

public:

    ~FrameWriter() { };

    void initialize(const std::string &source_name,
                    const oat::Frame &f) override;

    void write(void) override;
    
private:

    cv::VideoWriter video_writer_; 

};
}      /* namespace oat */
#endif /* OAT_FRAMEWRITER_H */
