//******************************************************************************
//* File:   PositionSocket.cpp
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

#include "PositionSocket.h"

#include <string>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/Source.h"

namespace oat {

PositionSocket::PositionSocket(const std::string &position_source_address)
: name_("posisock[" + position_source_address + "->*]")
, position_source_address_(position_source_address)
{
    // Nothing
}

bool PositionSocket::connectToNode()
{
    // Establish our a slot in the node 
    position_source_.touch(position_source_address_);

    // Wait for synchronous start with sink when it binds its node
    if (position_source_.connect() != SourceState::CONNECTED)
        return false;

    return true;
}

int PositionSocket::process()
{
    // START CRITICAL SECTION //
    ////////////////////////////
    node_state_ = position_source_.wait();
    if (node_state_ == oat::NodeState::END)
        return 1;

    // Clone the shared position
    internal_position_ = position_source_.clone();

    // Tell sink it can continue
    position_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Send the newly acquired position
    sendPosition(internal_position_);

    // Sink was not at END state
    return 0;
}

} /* namespace oat */
