//******************************************************************************
//* File:   TestPosition.cpp
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

#include "PoseGenerator.h"

#include <chrono>
#include <string>
#include <thread>

#include <cpptoml.h>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

PoseGenerator::PoseGenerator(const std::string &pose_sink_address)
: pose_sink_(pose_sink_address)
{
    tick_ = clock_.now();
    set_name("*",pose_sink_address);
}

po::options_description PoseGenerator::baseOptions(void) const
{
    po::options_description base_opts;

    // Common program options
    base_opts.add_options()
        ("rate,r", po::value<double>(),
         "Samples per second. Defaults to as fast as possible.")
        ("num-samples,n", po::value<uint64_t>(),
         "Number of position samples to generate and serve. Deafaults to "
         "approximately infinite.")
        ("unit-of-length,u", po::value<int>(),
         "Unit of legth in which generated pose position is specified: "
         "Values:\n"
         "  0:  \tPixels (default).\n"
         "  1:  \tMeters")
        ("room,R", po::value<std::string>(),
         "Array of floats, [x0,w,y0,l,z0,h], specifying the boundaries in "
         "which generated poses reside. The room has periodic boundaries so "
         "when a pose's position leaves one side it will enter the opposing "
         "one.")
        ;

    return base_opts;
}

bool PoseGenerator::connectToNode()
{
    // Bind to sink sink node and create a shared position
    pose_sink_.bind(sample_period_);

    start_ = clock_.now();
    return true;
}

int PoseGenerator::process()
{
    auto time_since_start
        = std::chrono::duration_cast<oat::Token::Microseconds>(clock_.now()
                                                               - start_);
    // Generate a pose and increment the sample count
    oat::Pose pose(sample_period_);
    bool eof = generate(pose, time_since_start, it_++);

    // Synchronously push the position and enforce sample clock if needed
    if (!eof) {

        pose_sink_.push(std::move(pose));

        if (enforce_sample_clock_) {
            auto tock = clock_.now();
            std::this_thread::sleep_for(sample_period_ - (tock - tick_));
            tick_ = clock_.now();
        }
    }

    return eof;
}


} /* namespace oat */
