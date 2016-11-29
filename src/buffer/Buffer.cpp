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

int Buffer::control(const char *msg)
{
    char id[32];
    identity(id, 32);
    std::cout << "[" << id << "] received: " << msg << std::endl;

    return 0; // Continue
}

//po::options_description Buffer::options()
//{
//    po::options_description local_opts;
//    local_opts.add_options()
//        ("buffer-size,n", po::value<uint64_t>(),
//        "Maximal size of buffer before overflow. Defaults to 1000.")
//        ;
//
//    return local_opts;
//}
//
//void Buffer::applyConfiguratio(const po::variables_map &vm)
//{
//    // Buffer size
//    size_t n;
//    if (oat::config::getNumericValue<size_t>(vm, config_table, "buffer-size", n, 0))
//        std::cout << "Setting buffer size not implemented.\n";
//}
} /* namespace oat */
