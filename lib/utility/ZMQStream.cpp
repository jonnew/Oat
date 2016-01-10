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
//******************************************************************************

#include <algorithm>
#include <cstring>
#include <iosfwd>
#include <zmq.hpp>

#include "ZMQStream.h"

namespace oat {

zmq_istream::zmq_istream(const p_zmq_context context,
                         const p_zmq_socket socket) :
  context_(context)
, socket_(socket)
//, index_(buffer_.size())
{
    // Nothing
}

std::streamsize zmq_istream::read(char *s, std::streamsize n) {

    zmq::message_t message;

    if (socket_->recv(&message)) {
        
        std::streamsize actual_n = 
            std::min(n, static_cast<std::streamsize>(message.size()));
        memcpy(s, static_cast<char *>(message.data()), actual_n);
        return actual_n;

    } else {

        return -1; //EOF
    }
}

zmq_ostream::zmq_ostream(const p_zmq_context context,
                         const p_zmq_socket socket) :
  context_(context)
, socket_(socket)
{
    // Nothing
}

std::streamsize zmq_ostream::write(const char *s, std::streamsize n) {

    zmq::message_t message(n);
    memcpy(static_cast<void *>(message.data()), s, n);
    return socket_->send(message) ? n : -1;
}
} /* namespace oat */
