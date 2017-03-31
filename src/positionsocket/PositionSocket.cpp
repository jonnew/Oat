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

namespace oat {

PositionSocket::PositionSocket(const std::string &pose_source_addresss)
: name_("posisock[" + pose_source_addresss + "->*]")
, pose_source_(pose_source_addresss)
{
    // Nothing
}

bool PositionSocket::connectToNode()
{
    // Wait for synchronous start with sink when it binds its node
    auto rc = (pose_source_.connect() == SourceState::connected);
    return rc;
}

int PositionSocket::process()
{
    // Synchronous pull from source
    oat::Pose pose;
    auto rc = pose_source_.pull(pose);
    if (rc) { return rc; }

    // Process the newly acquired pose
    sendPosition(pose);

    // Return sink state
    return rc;
}

} /* namespace oat */
