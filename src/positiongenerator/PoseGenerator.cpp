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

#include <chrono>
#include <string>
#include <thread>

#include <cpptoml.h>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"

#include "PoseGenerator.h"

namespace oat {

PoseGenerator::PoseGenerator(const std::string &pose_sink_address)
: name_("posigen[*->" + pose_sink_address + "]")
, pose_sink_address_(pose_sink_address)
{
    tick_ = clock_.now();
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
    pose_sink_.bind(pose_sink_address_);
    shared_pose_ = pose_sink_.retrieve();

    // Setup sample rate info on internal copy
    shared_pose_->set_rate_hz(1.0 / sample_period_in_sec_.count());

    return true;
}

int PoseGenerator::process()
{
    // Generate a pose
    oat::Pose pose;
    bool eof = generatePosition(pose);

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    pose_sink_.wait();

    if (first_pos_) {
        first_pos_ = false;
        start_ = clock_.now();
    }

    // Pure SINKs update pose and increment sample count
    auto time_since_start = std::chrono::duration_cast<Sample::Microseconds>(
        clock_.now() - start_);
    shared_pose_->produce(pose, time_since_start);

    // Tell sources there is new data
    pose_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    if (enforce_sample_clock_) {
        auto tock = clock_.now();
        std::this_thread::sleep_for(sample_period_in_sec_ - (tock - tick_));
        tick_ = clock_.now();
    }

    return eof;
}

void PoseGenerator::generateSamplePeriod(const double samples_per_second)
{
    oat::Sample::Seconds period(1.0 / samples_per_second);
    sample_period_in_sec_ = period; // Auto conversion
}

} /* namespace oat */
