//******************************************************************************
//* File:   FrameFilter.cpp
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
//******************************************************************************

#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/SharedFrameHeader.h"

#include "FrameFilter.h"

namespace oat {

FrameFilter::FrameFilter(const std::string &frame_source_address,
                         const std::string &frame_sink_address) :
  name_("framefilt[" + frame_source_address + "->" + frame_sink_address + "]")
, frame_source_address_(frame_source_address)
, frame_sink_address_(frame_sink_address)
{
  // Nothing
}

void FrameFilter::connectToNode() {

    // Connect to source node and retrieve cv::Mat parameters
    frame_source_.connect(frame_source_address_);
    frame_source_.verify();

    // Get frame meta data to format sink
    oat::Source<oat::SharedFrameHeader>::ConnectionParameters param =
            frame_source_.parameters();

    // Bind to sink sink node and create a shared cv::Mat
    frame_sink_.bind(frame_sink_address_, param.bytes);
    shared_frame_ = frame_sink_.retrieve(param.rows, param.cols, param.type);
}

bool FrameFilter::processFrame() {

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    node_state_ = frame_source_.wait();
    if (node_state_ == oat::NodeState::END)
        return true;

    // Clone the shared frame
    frame_source_.copyTo(internal_frame_);

    // Tell sink it can continue
    frame_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Mess with internal frame
    filter(internal_frame_);

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    frame_sink_.wait();

    internal_frame_.copyTo(shared_frame_);

    // Tell sources there is new data
    frame_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Sink was not at END state
    return false;
}

} /* namespace oat */
