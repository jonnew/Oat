//******************************************************************************
//* File:   UDPClient.cpp
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
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

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/rapidjson/rapidjson.h"

#include "SocketWriteStream.h"
#include "UDPClient.h"

UDPClient::UDPClient(const std::string& position_source_name, const std::string& host, const std::string& port) :
  PositionSocket(position_source_name)
, host_(host)
, port_(port)
, socket_(io_service, UDPEndpoint(boost::asio::ip::udp::v4(), 0)) { 

    UDPResolver resolver(io_service);
    UDPEndpoint endpoint = *resolver.resolve({boost::asio::ip::udp::v4(), host_, port_});
    
    upd_stream_.reset(new rapidjson::SocketWriteStream<UDPSocket, UDPEndpoint>(
            &socket_, endpoint, buffer_, sizeof(buffer_)));
    udp_writer_.Reset(*upd_stream_);
    
    udp_writer_.StartObject();
}

UDPClient::~UDPClient() {

    udp_writer_.EndObject();
    upd_stream_->Flush();
}

void UDPClient::servePosition(const oat::Position2D& current_position, const uint32_t sample) {
    
    // TODO: Sample should be a data member of position type!
    std::string sample_str = std::to_string(sample);
#ifdef RAPIDJSON_HAS_STDSTRING
            udp_writer_.String(sample_str);
#else
            udp_writer_.String(sample_str.c_str(), 
                    (rapidjson::SizeType)sample_str.length());
#endif
    current_position.Serialize(udp_writer_);
}

