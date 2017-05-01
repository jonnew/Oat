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

#include "FrameFilter.h"

#include <string>

namespace oat {

FrameFilter::FrameFilter(const std::string &frame_source_address,
                         const std::string &frame_sink_address)
: Component()
, frame_source_(frame_source_address)
, frame_sink_(frame_sink_address)
{
    // Set the component name for this instance
    set_name(frame_source_address, frame_sink_address);
}

bool FrameFilter::connectToNode()
{
    // Wait for synchronous start with sink when it binds its node
    if (frame_source_.connect() != SourceState::connected)
        return false;

    // Get temp frame
    auto tmp = frame_source_.retrieve();

    // Check the pixel color
    if (!checkPixelColor(tmp->color())) {
        throw std::runtime_error(
            "Source provides frames with incompatible pixel type.");
    }

    // Bind to sink node and create a shared frame
    frame_sink_.reserve(tmp->bytes());
    frame_sink_.bind(tmp->period(), tmp->rows(), tmp->cols(), tmp->color());

    return true;
}

int FrameFilter::process()
{
    // Synchronous pull from source
    oat::Frame frame;
    auto rc = frame_source_.pull(frame);
    if (rc) { return rc; }

    // Filter the frame and push
    filter(frame);

    // Move to  shared frame
    frame_sink_.push(std::move(frame));

    return rc;
}

} /* namespace oat */
