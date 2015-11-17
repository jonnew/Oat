//******************************************************************************
//* File:   UDPServer.h
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

#ifndef OAT_UDPSERVER_H
#define	OAT_UDPSERVER_H

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>

#include <rapidjson/rapidjson.h>

#include "SocketWriteStream.h"
#include "PositionSocket.h"

namespace oat {

// Forward decl.
class Position2D;

class UDPPositionServer : public PositionSocket {

    using UDPSocket = boost::asio::ip::udp::socket;
    using UDPEndpoint = boost::asio::ip::udp::endpoint;
    using UDPResolver = boost::asio::ip::udp::resolver;
    using DeadlineTimer = boost::asio::deadline_timer;

public:

    /**
     *
     * @param position_source_name
     * @param port
     */
    UDPPositionServer(const std::string& position_source_name,
             const unsigned short port);

private:

    // RX/TX buffers
    static const size_t MAX_LENGTH {65507}; // max udp buffer size
    char tx_buffer_[MAX_LENGTH]; // Buffer is flushed after each position read
    char rx_buffer_[MAX_LENGTH];

    // UDP communication specs
    UDPSocket socket_;
    UDPEndpoint endpoint_;
    DeadlineTimer input_deadline_;

    // Custom RapidJSON UDP stream and writer
    std::unique_ptr < rapidjson::SocketWriteStream
                    < UDPSocket, UDPEndpoint > > udp_stream_;

    /**
     *
     * @param position
     * @param sample
     */
    void sendPosition(const oat::Position2D& position) override;

};

}      /* namespace oat */
#endif /* OAT_UDPSERVER_H */

