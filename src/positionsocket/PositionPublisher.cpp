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

#include "PositionPublisher.h"

#include <string>
#include <zmq.hpp>

#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

PositionPublisher::PositionPublisher(const std::string &position_source_address)
: PositionSocket(position_source_address)
, publisher_(context_, ZMQ_PUB)
{
    // Nothing
}

void PositionPublisher::appendOptions(po::options_description &opts)
{
    // Accepts a config file
    PositionSocket::appendOptions(opts);

    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("endpoint,e", po::value<std::string>(),
         "ZMQ-style endpoint: '<transport>://<host>:<port>'. For instance, "
         "'tcp://*:5555' or 'ipc://*:5556' specify TCP and interprocess "
         "communication on ports 5555 or 5556, respectively")
        ;
    opts.add(local_opts);

    // Return valid keys
    for (auto &o: local_opts.options())
        config_keys_.push_back(o->long_name());
}

void PositionPublisher::configure(const po::variables_map &vm)
{
    // Check for config file and entry correctness. In this case, make sure
    // that none have been provided
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Endpoint
    std::string endpoint;
    oat::config::getValue<std::string>(vm, config_table, "endpoint", endpoint, true);
    publisher_.bind(endpoint);
}

void PositionPublisher::sendPosition(const oat::Position2D &position)
{
    // Serialize the current position
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    position.Serialize(writer);

    // Publish update
    zmq::message_t zmsg(buffer.GetSize());
    memcpy((void *)zmsg.data(), buffer.GetString(), buffer.GetSize());
    publisher_.send(zmsg);
}

} /* namespace oat */
