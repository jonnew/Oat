//******************************************************************************
//* File:   ZMQStream.cpp
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
//*****************************************************************************

#include <cstring.h>
#include <iostream>
#include <zmq.h>

#include "ZMQStream.h"

namespace oat {

zmq_istream::zmq_istream(const std::string &endpoint) :
  socket_(context_, ZMQ_SUB)
, index(buffer_.size())
{

    socket_.connect(endpoint.c_str());
    socket_.setsocketopt(ZMQ_SUBSCRIBE, "", 0);
}

std::streamsize zmq_stream::read(char *s, std::streamsize n) {

    zmq::message_t message;

    if (socket.recv(&message)) {

        if (index_ != buffer_.size()) {

            auto size = std::min(buffer_.size() - index_, static_cast<buffer_size_t>(n));
            memcpy(s, &buffer_[index_], size);
            index_ += size;
            return size;

        } else if (message.size() < static_cast<buffer_size_t>(n)) {

            memcpy(s, (const char *)message.data(), message.size());
            return message.size();

        } else {

            memcpy(s, (const char *)message.data(), n);
            buffer_.resize(message.size() - n); // TODO do not resize if smaller ( for performance )
            memcpy(&buffer_[0], (const char *)message.data(), buffer_.size());
            index_ = 0;
            return n;
        }

    } else {

        // EOF
        return -1;
    }
}

} /* namespace oat */

