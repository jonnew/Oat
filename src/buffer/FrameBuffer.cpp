//******************************************************************************
//* File:   FrameBuffer.cpp
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

#include "FrameBuffer.h"

#include <iostream>

namespace oat {

FrameBuffer::FrameBuffer(const std::string &source_address,
                         const std::string &sink_address)
: Buffer(source_address, sink_address)
{
    // Nothing
}

bool FrameBuffer::connectToNode()
{
    // Establish our a slot in the node
    source_.touch(source_address_);

    // Wait for sychronous start with sink when it binds the node
    if (source_.connect() != SourceState::CONNECTED)
        return false;

    // Get frame meta data to format sink
    auto param = source_.parameters();

    // Bind sink node
    sink_.bind(sink_address_, param.bytes);
    shared_frame_
        = sink_.retrieve(param.rows, param.cols, param.type, param.color);

    // Start consumer thread
    sink_thread_ = std::thread(&FrameBuffer::pop, this);

    return true;
}

int FrameBuffer::process()
{
    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    if (source_.wait() == oat::NodeState::END)
        return 1;

    if (!buffer_.push(source_.clone()))
        std::cerr << "Buffer overrun.\n";

    // Tell sink it can continue
    source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Notify comsumer thread that it can proceed
    cv_.notify_one();

#ifndef NDEBUG
    showBufferState(buffer_, BUFFSIZE);
#endif

    // Sink was not at END state
    return 0;
}

void FrameBuffer::pop()
{
    while (sink_running_) {

        // Proceed only if buffer_ has data
        std::unique_lock<std::mutex> lk(cv_m_);
        if  (cv_.wait_for(lk, msec(10)) == std::cv_status::timeout)
            continue;

        // Publish objects when they are requested until the buffer
        // is empty
        while (buffer_.read_available() > 0) {

            // START CRITICAL SECTION //
            ////////////////////////////

            // Wait for sources to read
            sink_.wait();

            // TODO: use specialized spsc allocator for popping somehow?
            buffer_.consume_one(
                [this](oat::Frame frame){ frame.copyTo(shared_frame_); }
            );

            // Tell sources there is new data
            sink_.post();

            ////////////////////////////
            //  END CRITICAL SECTION  //
        }
    }
}

} /* namespace oat */
