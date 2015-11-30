//******************************************************************************
//* File:   TokenBuffer.h
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

#ifndef OAT_TOKEN_BUFFER_H
#define	OAT_TOKEN_BUFFER_H

#include <boost/lockfree/spsc_queue.hpp>

#include "../../lib/datatypes/Position2D.h"

#include "Buffer.h"

namespace oat {

/**
 * Generic token buffer.
 */
template <typename T>
class TokenBuffer : public Buffer {

    using SPSCBuffer =
        boost::lockfree::spsc_queue<T, buffer_size_t>;

public:

    /**
     * Generic token buffer.
     *
     * @param source_address SOURCE node address
     * @param sink_address SINK node address
     */
    TokenBuffer(const std::string &source_address,
                const std::string &sink_address);

    /**
     * Buffers must be able to connect to SOURCE and SINK nodes in shared
     * memory.
     */
    void connectToNode(void) override;

    ///**
    // * Obtain new token from SOURCE and push onto FIFO.
    // * @return SOURCE end-of-stream signal. If true, this component should exit.
    // */
    bool push(void) override;

private:

    /**
     * In response to downstream request, publish tokens from FIFO to SINK.
     */
    void pop(void) override;

    // Source
    oat::Source<T> source_;

    // Buffer
    SPSCBuffer buffer_;

    // Sink
    T * shared_token_;
    oat::Sink<T> sink_;
};

}      /* namespace oat */
#endif /* OAT_TOKEN_BUFFER_H */
