//******************************************************************************
//* File:   PositionCout.cpp
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

#include "PositionCout.h"

#include <iostream>
#include <string>

#include <rapidjson/rapidjson.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

PositionCout::PositionCout(const std::string &position_source_address)
: PositionSocket(position_source_address)
{
    // Nothing
}

void PositionCout::appendOptions(po::options_description &opts) {

    // Accepts a config file
    PositionSocket::appendOptions(opts);

    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("pretty-print,p", 
         "If true, print formated positions to the command line.")
        ;

    opts.add(local_opts);

    // Return valid keys
    for (auto &o: local_opts.options())
        config_keys_.push_back(o->long_name());
}

void PositionCout::configure(const po::variables_map &vm) {

    // Check for config file and entry correctness. In this case, make sure
    // that none have been provided
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Timestamp
    oat::config::getValue<bool>(vm, config_table, "pretty-print", pretty_);
}

void PositionCout::sendPosition(const oat::Position2D &position)
{
    // Serialize the current position
    rapidjson::StringBuffer buffer;

    if (pretty_) {
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        position.Serialize(writer);
    } else {
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        position.Serialize(writer);
    }

    std::cout << buffer.GetString() << std::flush;
}

} /* namespace oat */
