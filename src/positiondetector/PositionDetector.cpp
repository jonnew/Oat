//******************************************************************************
//* File:   PositionDetector.cpp
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

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/SharedCVMat.h"

#include "PositionDetector.h"

namespace oat {

PositionDetector::PositionDetector(const std::string &frame_source_address,
                                   const std::string &position_sink_address) :
  name_("posidet[" + frame_source_address + "->" + position_sink_address + "]")
, frame_source_address_(frame_source_address)
, position_sink_address_(position_sink_address)
{
  // Nothing
}

void PositionDetector::connectToNode() {

    // Connect to source node and retrieve cv::Mat parameters
    frame_source_.connect(frame_source_address_);

    // Bind to sink node and create a shared position
    position_sink_.bind(position_sink_address_, position_sink_address_);
    shared_position_ = position_sink_.retrieve();
}

bool PositionDetector::process() {

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    if (frame_source_.wait() == oat::NodeState::END)
        return true;

    // Clone the shared frame
    internal_frame_ = frame_source_.clone();

    // Tell sink it can continue
    frame_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Mess with internal frame
    detectPosition(internal_frame_, internal_position_);

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    position_sink_.wait();

    *shared_position_ = internal_position_;

    // Tell sources there is new data
    position_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Sink was not at END state
    return false;
}

} /* namespace oat */
