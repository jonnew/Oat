//******************************************************************************
//* File:   UDPServer.cpp
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

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>
#include <boost/asio/deadline_timer.hpp>

#include <rapidjson/rapidjson.h>

#include "../../lib/datatypes/Position2D.h"

#include "SocketWriteStream.h"
#include "UDPPositionServer.h"

namespace oat {

UDPPositionServer::UDPPositionServer(const std::string &position_source_address,
                                     const unsigned short port) :
  PositionSocket(position_source_address)
, socket_(io_service_, UDPEndpoint(boost::asio::ip::udp::v4(), port))
, input_deadline_(io_service_) {

    input_deadline_.expires_at(boost::posix_time::pos_infin);

    udp_stream_.reset(new rapidjson::SocketWriteStream<UDPSocket, UDPEndpoint>(
            &socket_, endpoint_, tx_buffer_, sizeof(tx_buffer_)));

}

void UDPPositionServer::sendPosition(const oat::Position2D& current_position) {

    rapidjson::Writer < rapidjson::SocketWriteStream
                      < UDPSocket, UDPEndpoint > > udp_writer_ {*udp_stream_};

    // Need to receive request from remote client in order to proceed.
    // TODO: check request message contents?
    // TODO: A request does not nessesarily result in a full positional datum being sent, but just
    //       in the current position being serialized and maybe sent if flush is called within the
    //       the SocktWriteStream.
    // TODO: This operation needs a timeout to prevent everything from locking up
    //       the program wants to quit but nobody is sending any requests.

    /*size_t length = */ socket_.receive_from(
        boost::asio::buffer(rx_buffer_, MAX_LENGTH), endpoint_);

    current_position.Serialize(udp_writer_);

    // Flush the stream after each Serialization call so that each UDP packet
    // corresponds to a single position value
    udp_stream_->Flush();
}

} /* namespace oat */