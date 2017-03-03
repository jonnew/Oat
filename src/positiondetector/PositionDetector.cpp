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

#include "PositionDetector.h"

#include <string>

#include <opencv2/core/mat.hpp>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/Source.h"

namespace oat {

PositionDetector::PositionDetector(const std::string &frame_source_address,
                                   const std::string &pose_sink_address)
: name_("posidet[" + frame_source_address + "->" + pose_sink_address + "]")
, frame_source_address_(frame_source_address)
, pose_sink_address_(pose_sink_address)
{
  // Nothing
}

bool PositionDetector::connectToNode()
{
    // Establish our a slot in the node
    frame_source_.touch(frame_source_address_);

    // Wait for synchronous start with sink when it binds its node
    if (frame_source_.connect(required_color_) != SourceState::CONNECTED)
        return false;

    // Bind to sink node and create a shared position
    pose_sink_.bind(pose_sink_address_);
    shared_pose_ = pose_sink_.retrieve();

    return true;
}

int PositionDetector::process()
{
    oat::Frame frame;

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    if (frame_source_.wait() == oat::NodeState::END)
        return 1;

    // Clone the shared frame
    frame_source_.copyTo(frame);

    // Tell sink it can continue
    frame_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Propagate sample info and detect position
    auto pose = detectPose(frame);

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    pose_sink_.wait();

    *shared_pose_ = pose;

    // Tell sources there is new data
    pose_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Sink was not at END state
    return 0;
}

} /* namespace oat */
