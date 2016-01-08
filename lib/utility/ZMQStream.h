//******************************************************************************
//* File:   ZMQStream.h
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

#ifndef OAT_ZMQSTREAM_H
#define OAT_ZMQSTREAM_H

#include <iosfwd>
#include <boost/iostreams/concepts.hpp>
#include <memory>
#include <zmq.hpp>

namespace oat {

namespace io = boost::iostreams;

using p_zmq_context = std::shared_ptr<zmq::context_t>;
using p_zmq_socket = std::shared_ptr<zmq::socket_t>;

class zmq_istream : public io::source {

    //using buffer_t = std::vector<char>;
    //using buffer_size_t = buffer_t::size_type;

public:

    /**
     * ZMQ input stream.
     *
     * @param context zeromq context that facilitates this stream
     * @parame socket zeromq socket to receive messages from
     */
    zmq_istream(const p_zmq_context context,
                const p_zmq_socket socket);

    std::streamsize read(char *s, std::streamsize n);
    zmq::socket_t & socket() { return *socket_; }

private:

    // Need these since zmq contexts and sockets are not copy-constructable
    const p_zmq_context context_;
    const p_zmq_socket socket_;

    // TODO: Needed for long messages, but not implemented right now
    //buffer_t buffer_;
    //buffer_size_t index_;
};

class zmq_ostream : public io::sink {

public:

    /**
     * ZMQ output stream.
     *
     * @param context zeromq context that facilitates this stream
     * @parame socket zeromq socket to psuh messages to
     */
    zmq_ostream(const p_zmq_context context,
                const p_zmq_socket socket);

    std::streamsize write(const char *s, std::streamsize n);
    zmq::socket_t & socket() { return *socket_; }

private:

    // Need these since zmq contexts and sockets are not copy-constructable
    const p_zmq_context context_;
    const p_zmq_socket socket_;
};
}      /* namespace oat */
#endif /* OAT_ZMQSTREAM_H */

