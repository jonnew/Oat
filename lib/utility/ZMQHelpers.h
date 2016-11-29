//******************************************************************************
//* File:   ZMQHelpers.h
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

#pragma once

#include <zmq.hpp>

namespace oat {

inline std::string recvString(zmq::socket_t *socket)
{
    zmq::message_t message;
    socket->recv(&message);

    return std::string(static_cast<char *>(message.data()), message.size());
}

inline bool sendString(zmq::socket_t *socket, const std::string &string)
{
    zmq::message_t message(string.size());
    memcpy(message.data(), string.data(), string.size());

    bool rc = socket->send(message);
    return rc;
}

inline bool sendStringMore(zmq::socket_t *socket, const std::string &string)
{
    zmq::message_t message(string.size());
    memcpy(message.data(), string.data(), string.size());

    bool rc = socket->send(message, ZMQ_SNDMORE);
    return rc;
}

inline bool sendReqEnvelope(zmq::socket_t *socket,
                           const std::string &id,
                           const std::string &data)
{
    bool good = true;
    good &= sendStringMore(socket, id);
    good &= sendStringMore(socket, "");
    good &= sendString(socket, data);

    return good;
}

inline bool recvReqEnvelope(zmq::socket_t *socket,
                           std::string &id,
                           std::string &data)
{
   id = recvString(socket);
   recvString(socket);
   data = recvString(socket);

   if (id.empty() || data.empty())
       return false;
   else
       return true;
}
} /* namespace oat */
