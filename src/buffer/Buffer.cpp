//******************************************************************************
//* File:   Buffer.cpp
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

#include "Buffer.h"

#include <string>

namespace oat {

Buffer::Buffer(const std::string &source_address,
               const std::string &sink_address)
: Component()
, name_("buffer[" + source_address + "->" + sink_address + "]")
, source_address_(source_address)
, sink_address_(sink_address)
{
    // Nothing
}

Buffer::~Buffer()
{
    // Join threads
    sink_running_ = false;
    if (sink_thread_.joinable())
        sink_thread_.join();
}

po::options_description Buffer::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("down-sample-factor,d", po::value<size_t>(),
         "Positive integer value, specifying the token decimation factor of the "
         "buffer. The outgoing stream's sample rate will be the incoming stream's rate "
         "divided by this number. Defaults to 1.")
        ;

    return local_opts;
}

void Buffer::applyConfiguration(const po::variables_map &vm,
                                const config::OptionTable &config_table)
{
    // Resample ratio
    oat::config::getNumericValue<size_t>(
        vm, config_table, "down-sample-factor", down_sample_factor_, 1);
}

} /* namespace oat */
