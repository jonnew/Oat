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
#include <cassert>

#include <opencv2/core/mat.hpp>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/shmemdf/Sink2.h"
#include "../../lib/shmemdf/Source.h"

namespace oat {

PositionDetector::PositionDetector(const std::string &frame_source_address,
                                   const std::string &pose_sink_address)
: name_("posidet[" + frame_source_address + "->" + pose_sink_address + "]")
, frame_source_(frame_source_address)
, pose_sink_(pose_sink_address)
{
  // Nothing
}

bool PositionDetector::connectToNode()
{
    // Wait for synchronous start with sink when it binds its node
    if (frame_source_.connect() != SourceState::connected)
        return false;
    
    // Check the pixel color
    if (!checkPixelColor(frame_source_.retrieve()->color())) {
        throw std::runtime_error(
            "Source provides frames with incompatible pixel type.");
    }

    // Bind to sink node and create a shared position
    pose_sink_.bind();

    return true;
}

int PositionDetector::process()
{
    // Synchronous pull from source
    oat::SharedFrame *sh_frame;
    auto rc = frame_source_.pull(&sh_frame);
    if (rc) { return rc; }

    // Process the newly acquired frame and push result to sink
    auto frame = oat::frame::getLocal(*sh_frame);
    pose_sink_.push(detectPose(frame));

    return rc;
}

} /* namespace oat */
