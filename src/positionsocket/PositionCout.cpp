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

po::options_description PositionCout::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("pretty-print,p", 
         "If true, print formated positions to the command line.")
        ;

    return local_opts; 
}

void PositionCout::applyConfiguration(const po::variables_map &vm,
                                      const config::OptionTable &config_table)
{
    // Format output
    oat::config::getValue<bool>(vm, config_table, "pretty-print", pretty_);
}

void PositionCout::sendPosition(const oat::Position2D &position)
{
    // Serialize the current position
    rapidjson::StringBuffer buffer;

    if (pretty_) {
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        oat::serializePosition(position, writer);
    } else {
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        oat::serializePosition(position, writer);
    }

    std::cout << buffer.GetString() << std::flush;
}

} /* namespace oat */
