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

#include "PositionDetector.h"

namespace oat {

PositionDetector::PositionDetector(const std::string &frame_source_address,
                                   const std::string &position_sink_address)
: name_("posidet[" + frame_source_address + "->" + position_sink_address + "]")
, frame_source_address_(frame_source_address)
, position_sink_address_(position_sink_address)
{
  // Nothing
}

void PositionDetector::appendOptions(po::options_description &opts)
{
    // Common program options
    opts.add_options()
        ("config,c", po::value<std::vector<std::string> >()->multitoken(),
        "Configuration file/key pair.\n"
        "e.g. 'config.toml mykey'")
        ;
}

void PositionDetector::connectToNode()
{
    // Establish our a slot in the node
    frame_source_.touch(frame_source_address_);

    // Wait for synchronous start with sink when it binds the node
    frame_source_.connect(required_color_);

    // Bind to sink node and create a shared position
    position_sink_.bind(position_sink_address_, position_sink_address_);
    shared_position_ = position_sink_.retrieve();
}

bool PositionDetector::process()
{
    oat::Frame internal_frame;
    oat::Position2D internal_pos("");

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    if (frame_source_.wait() == oat::NodeState::END)
        return true;

    // Clone the shared frame
    frame_source_.copyTo(internal_frame);

    // Tell sink it can continue
    frame_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Propagate sample info and detect position
    internal_pos.set_sample(internal_frame.sample());
    detectPosition(internal_frame, internal_pos);

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    position_sink_.wait();

    *shared_position_ = internal_pos;

    // Tell sources there is new data
    position_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Sink was not at END state
    return false;
}

} /* namespace oat */
