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
: name_("buffer[" + source_address + "->" + sink_address + "]")
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

} /* namespace oat */
