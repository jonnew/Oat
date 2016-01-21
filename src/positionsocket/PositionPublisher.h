//******************************************************************************
//* File:   PositionPublisher.h
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

#ifndef OAT_POSITIONPUBLISHER_H
#define	OAT_POSITIONPUBLISHER_H

#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>
#include <zmq.hpp>

#include "SocketWriteStream.h"
#include "PositionSocket.h"

namespace oat {

// Forward decl.
class Position2D;

class PositionPublisher : public PositionSocket {

public:
    PositionPublisher(const std::string &position_source_address,
                      const std::string &endpoint);

private:

    // Custom RapidJSON UDP stream
    static const size_t MAX_LENGTH {65507}; // max udp buffer size
    char buffer_[MAX_LENGTH]; // Buffer is flushed after each position read

    zmq::socket_t publisher_;

    //std::unique_ptr < rapidjson::SocketWriteStream
    //                < UDPSocket, UDPEndpoint > > udp_stream_;

    void sendPosition(const oat::Position2D& position) override;
};

}      /* namespace oat */
#endif /* OAT_POSITIONPUBLISHER_H */
