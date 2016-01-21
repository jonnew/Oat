//******************************************************************************
//* File:   PositionPublisher.cpp
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
#include <zmq.hpp>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"

#include "PositionPublisher.h"

namespace oat {

PositionPublisher::PositionPublisher(const std::string &position_source_address,
                                     const std::string &endpoint) :
  PositionSocket(position_source_address)
, publisher_(context_, ZMQ_PUB)
{
    publisher_.bind(endpoint);
}

void PositionPublisher::sendPosition(const oat::Position2D& position) {

    // Serialize the current position
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    position.Serialize(writer);

    // Publish update
    zmq::message_t zmsg(buffer.GetSize()); 
    memcpy((void *)zmsg.data(), buffer.GetString(), buffer.GetSize());
    publisher_.send(zmsg);
    
    //std::cout << "Sending " << (char *)(zmsg.data()) << "\n"; 
}

} /* namespace oat */
