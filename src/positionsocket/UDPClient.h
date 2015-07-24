//******************************************************************************
//* File:   UDPClient.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
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
#ifndef UDPSERVER_H
#define	UDPSERVER_H

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>

#include "../../lib/rapidjson/rapidjson.h"

#include "SocketWriteStream.h"
#include "PositionSocket.h"

// Forward declarations
namespace oat { class Position2D; }

class UDPClient : public PositionSocket {

    using UDPSocket = boost::asio::ip::udp::socket;
    using UDPEndpoint = boost::asio::ip::udp::endpoint;
    using UDPResolver = boost::asio::ip::udp::resolver;
    
public:
    UDPClient(const std::string& position_source_name, 
              const std::string& host, 
              const std::string& port);

    ~UDPClient();
    
private:

    // Service object - binds to OS level io_service
    boost::asio::io_service io_service;
    
    // Address specification
    std::string host_;
    std::string port_; // TODO: What if user requests port less than 1000 without sudo?
    UDPSocket socket_;
    
    // Custom RapidJSON UDP stream
    char buffer_[10]; // TODO: This (1) May be redundant because boost::asio::buffer 
                      // is being used during the udp send_to call (2) If not,
                      // I need to make this so 1 position is sent every time
                      // a position is read from source. Implement at the level of
                      // SocketWriteStream?
    std::unique_ptr < rapidjson::SocketWriteStream
                    < UDPSocket, UDPEndpoint > > upd_stream_;
    rapidjson::Writer < rapidjson::SocketWriteStream 
                      < UDPSocket, UDPEndpoint > > udp_writer_ {*upd_stream_};
    
    void servePosition(const oat::Position2D& position, const uint32_t sample);
   
};

#endif	/* UDPSERVER_H */

