//******************************************************************************
//* File:   PositionReplier.cpp
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

#include "PositionReplier.h"

#include <string>
#include <zmq.hpp>

#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

PositionReplier::PositionReplier(const std::string &position_source_address)
: PositionSocket(position_source_address)
, replier_(context_, ZMQ_REP)
{
    // Nothing
}

po::options_description PositionReplier::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("endpoint,e", po::value<std::string>(),
         "ZMQ-style endpoint. For TCP: '<transport>://<host>:<port>'. For instance, "
         "'tcp://*:5555'. Or, for interprocess communication: "
         "'<transport>:///<user-named-pipe>. For instance "
         "'ipc:///tmp/test.pipe'.");
        ;

    return local_opts;
}

void PositionReplier::applyConfiguration(
    const po::variables_map &vm, const config::OptionTable &config_table)
{
    // Endpoint
    std::string endpoint;
    oat::config::getValue<std::string>(vm, config_table, "endpoint", endpoint, true);
    replier_.bind(endpoint);
}

void PositionReplier::sendPosition(const oat::Position2D& position)
{
    // Serialize the current position
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    oat::serializePosition(position, writer);

    //  Wait for next request from client
    // TODO: Use incoming string to decide which part of the position to send
    zmq::message_t request;
    replier_.recv(&request);

    // Publish update
    zmq::message_t zmsg(buffer.GetSize());
    memcpy((void *)zmsg.data(), buffer.GetString(), buffer.GetSize());
    replier_.send(zmsg);
}

} /* namespace oat */
