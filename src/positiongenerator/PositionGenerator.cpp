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

#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

#include "PositionGenerator.h"

namespace oat {

PositionGenerator::PositionGenerator(const std::string &position_sink_address)
: name_("posigen[*->" + position_sink_address + "]")
, position_sink_address_(position_sink_address)
{
    tick_ = clock_.now();
}

void PositionGenerator::appendOptions(po::options_description &opts)
{
    // Common program options
    opts.add_options()
        ("config,c", po::value<std::vector<std::string> >()->multitoken(),
        "Configuration file/key pair.\n"
        "e.g. 'config.toml mykey'")
        ("rate,r", po::value<double>(),
        "Samples per second. Defaults to as fast as possible.")
        ("num-samples,n", po::value<uint64_t>(),
        "Number of position samples to generate and serve. Deafaults to "
        "approximately infinite.")
        ("room,R", po::value<std::string>(),
         "Array of floats, [x0,y0,width,height], specifying the boundaries in "
         "which generated positions reside. The room has periodic boundaries so "
         "when a position leaves one side it will enter the opposing one.")
        ;
}

void PositionGenerator::connectToNode()
{
    // Bind to sink sink node and create a shared position
    position_sink_.bind(position_sink_address_, position_sink_address_);
    shared_position_ = position_sink_.retrieve();

    // Setup sample rate info on internal copy
    internal_position_.set_rate_hz(1.0 / sample_period_in_sec_.count());
}

bool PositionGenerator::process()
{
    // Generate internal position
    bool eof = generatePosition(internal_position_);

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    position_sink_.wait();

    if (first_pos_) {
        first_pos_ = false; 
        start_ = clock_.now();
    }

    *shared_position_ = internal_position_;

    // Tell sources there is new data
    position_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    if (enforce_sample_clock_) {
        auto tock = clock_.now();
        std::this_thread::sleep_for(sample_period_in_sec_ - (tock - tick_));
        tick_ = clock_.now();
    }

    // Pure SINKs increment sample count
    auto time_since_start = std::chrono::duration_cast<Sample::Microseconds>(
        clock_.now() - start_);
    internal_position_.incrementSampleCount(time_since_start);

    return eof;
}

void PositionGenerator::generateSamplePeriod(const double samples_per_second)
{
    oat::Sample::Seconds period(1.0 / samples_per_second);
    sample_period_in_sec_ = period; // Auto conversion
}

} /* namespace oat */
