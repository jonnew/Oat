//******************************************************************************
//* File:   PositionFilter.cpp
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

#include "PositionFilter.h"

namespace oat {

PositionFilter::PositionFilter(const std::string &position_source_address,
                               const std::string &position_sink_address) :
  name_("posifilt[" + position_source_address + "->" + position_sink_address + "]")
, position_source_address_(position_source_address)
, position_sink_address_(position_sink_address)
{
  // Nothing
}

void PositionFilter::connectToNode() {

    // Establish our a slot in the node 
    position_source_.touch(position_source_address_);

    // Wait for synchronous start with sink when it binds the node
    position_source_.connect();

    // Bind to sink sink node and create a shared position
    position_sink_.bind(position_sink_address_, position_sink_address_);
    shared_position_ = position_sink_.retrieve();
}

bool PositionFilter::process() {

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    if (position_source_.wait() == oat::NodeState::END)
        return true;

    // Clone the shared frame
    internal_position_ = position_source_.clone();

    // Tell sink it can continue
    position_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Mess with internal frame
    filter(internal_position_);

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
