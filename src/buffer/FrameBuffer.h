//******************************************************************************
//* File:   FrameBuffer.h
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

#ifndef OAT_FRAME_BUFFER_H
#define	OAT_FRAME_BUFFER_H

#include "Buffer.h"

#include <boost/lockfree/spsc_queue.hpp>

#include "../../lib/shmemdf/SharedFrameHeader.h"

namespace oat {

class FrameBuffer : public Buffer {

    using SPSCBuffer = boost::lockfree::spsc_queue<oat::Frame, buffer_size_t>;

public:
    /**
     * @brief Frame buffer.
     * @param source_address SOURCE node address
     * @param sink_address SINK node address
     */
    FrameBuffer(const std::string &source_address,
                const std::string &sink_address);

protected:
    // Implement Component interface
    bool connectToNode(void) override;
    int process(void) override;

private:
    void pop(void) override;

    // Source
    oat::Source<oat::Frame> source_;

    // Buffer
    SPSCBuffer buffer_;

    // Sink
    oat::Frame shared_frame_;
    oat::Sink<oat::Frame> sink_;
};

}      /* namespace oat */
#endif /* OAT_FRAME_BUFFER_H */
