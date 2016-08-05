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
#include <opencv2/cvconfig.h>
#include <opencv2/core/mat.hpp>

#include "../../lib/utility/make_unique.h"
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

void FrameFilter::appendOptions(po::options_description &opts) const {

    // Common program options
    opts.add_options()
        ("config,c", po::value<std::vector<std::string> >()->multitoken(),
        "Configuration file/key pair.\n"
        "e.g. 'config.toml mykey'")
        ;
}

void FrameFilter::connectToNode() {

    // Establish our a slot in the source node
    frame_source_.touch(frame_source_address_);

    // Wait for synchronous start with sink when it binds its node
    frame_source_.connect();

    // Get frame meta data to format sink
    frame_parameters_ = frame_source_.parameters();

    // Bind to sink node and create a shared frame
    frame_sink_.bind(frame_sink_address_, frame_parameters_.bytes);
    shared_frame_ = frame_sink_.retrieve(frame_parameters_.rows,
                                         frame_parameters_.cols,
                                         frame_parameters_.type);
}

bool FrameFilter::processFrame() {

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    if (frame_source_.wait() == oat::NodeState::END)
        return true;

    // Clone the shared frame
    frame_source_.copyTo(internal_frame_);

    // Tell sink it can continue
    frame_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Filter internal frame
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
