//******************************************************************************
//* File:   UDPClient.cpp
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

#include "UDPPositionClient.h"
#include "SocketWriteStream.h"

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>
#include <rapidjson/rapidjson.h>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

UDPPositionClient::UDPPositionClient(const std::string &position_source_address)
: PositionSocket(position_source_address)
, socket_(io_service_, UDPEndpoint(boost::asio::ip::udp::v4(), 0))
{
    // Nothing
}

void UDPPositionClient::appendOptions(po::options_description &opts)
{
    // Accepts a config file
    PositionSocket::appendOptions(opts);

    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("host,h", po::value<std::string>(),
         "Host IP address of remote device to send positions to. For "
         "instance, '10.0.0.1'.")
        ("port,p", po::value<int>(),
         "Port number of endpoint on remote device to send positions to. For "
         "instance, 5555.")
        ;
    opts.add(local_opts);

    // Return valid keys
    for (auto &o: local_opts.options())
        config_keys_.push_back(o->long_name());
}

void UDPPositionClient::configure(const po::variables_map &vm)
{
    // Check for config file and entry correctness. In this case, make sure
    // that none have been provided
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Host
    std::string host;
    oat::config::getValue<std::string>(
        vm, config_table, "host", host, true
    );

    // Port
    int port;
    oat::config::getNumericValue<int>(
        vm, config_table, "port", port, 1025, 65535, true
    );

    UDPResolver resolver(io_service_);
    UDPEndpoint endpoint = *resolver.resolve({boost::asio::ip::udp::v4(),
                                              host,
                                              std::to_string(port)});

    udp_stream_.reset(new rapidjson::SocketWriteStream<UDPSocket, UDPEndpoint>(
            &socket_, endpoint, buffer_, sizeof(buffer_)));
}

// Each position is sent in a single UDP packet
void UDPPositionClient::sendPosition(const oat::Position2D &current_position)
{
    rapidjson::Writer < rapidjson::SocketWriteStream
                      < UDPSocket, UDPEndpoint > > udp_writer_ {*udp_stream_};

    oat::serializePosition(current_position, udp_writer_);

    // Flush the stream after each Serialization call so that each UDP packet
    // corresponds to a single position value
    udp_stream_->Flush();
}

} /* namespace oat */
