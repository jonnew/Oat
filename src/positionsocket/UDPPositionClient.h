//******************************************************************************
//* File:   UDPClient.h
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

#ifndef OAT_UDPCLIENT_H
#define	OAT_UDPCLIENT_H

#include "PositionSocket.h"
#include "SocketWriteStream.h"

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>
#include <rapidjson/rapidjson.h>

namespace oat {

class UDPPositionClient : public PositionSocket {

    using UDPSocket = boost::asio::ip::udp::socket;
    using UDPEndpoint = boost::asio::ip::udp::endpoint;
    using UDPResolver = boost::asio::ip::udp::resolver;
    using SocketWriter = rapidjson::SocketWriteStream<UDPSocket, UDPEndpoint>;

public:
    UDPPositionClient(const std::string &pose_source_address);

private:
    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    // IO service
    boost::asio::io_service io_service_;
    UDPSocket socket_;

    // Custom RapidJSON UDP stream
    static constexpr size_t MAX_LENGTH {65507}; // max udp buffer size
    char buffer_[MAX_LENGTH]; // Buffer is flushed after each position read
    std::unique_ptr<SocketWriter> udp_stream_;

    void sendPosition(const oat::Pose& pose) override;
};

}      /* namespace oat */
#endif /* OAT_UDPCLIENT_H */
