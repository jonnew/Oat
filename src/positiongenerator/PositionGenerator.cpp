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

template <typename T>
PositionGenerator<T>::PositionGenerator(
    const std::string &position_sink_address)
: name_("posigen[*->" + position_sink_address + "]")
, position_sink_address_(position_sink_address)
{
    tick_ = clock_.now();
}

template <typename T>
PositionGenerator<T>::~PositionGenerator() 
{ 
    // Needed to prevent linking errors in derived classes
}

template <typename T>
void PositionGenerator<T>::appendOptions(po::options_description &opts)
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

template <typename T>
void PositionGenerator<T>::configure(const po::variables_map &vm)
{
    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);
    
    // Rate
    double fs = 10000;
    if (oat::config::getNumericValue<double>(
                vm, config_table, "rate", fs, 0)) {
        enforce_sample_clock_ = true;
    }
    generateSamplePeriod(fs);

    // Rate
    oat::config::getNumericValue<uint64_t>(
            vm, config_table, "num-samples", num_samples_, 0);

    // Room
    std::vector<double> r;
    if (oat::config::getArray<double, 4>(vm, config_table, "room", r)) {
        room_.x = r[0];
        room_.y = r[1];
        room_.width = r[2];
        room_.height = r[3];
    }
}
template <typename T>
void PositionGenerator<T>::connectToNode()
{
    // Bind to sink sink node and create a shared position
    position_sink_.bind(position_sink_address_, position_sink_address_);
    shared_position_ = position_sink_.retrieve();

    // Setup sample rate info on internal copy
    internal_position_.sample().set_rate_hz(1.0 / sample_period_in_sec_.count());
}

template <typename T>
bool PositionGenerator<T>::process()
{

    // Generate internal position
    bool eof = generatePosition(internal_position_);

    if (enforce_sample_clock_) {
        auto tock = clock_.now();
        std::this_thread::sleep_for(sample_period_in_sec_ - (tock - tick_));
        tick_ = clock_.now();
    }

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    position_sink_.wait();

    *shared_position_ = internal_position_;

    // Tell sources there is new data
    position_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Pure SINK so it needs to update sample count
    internal_position_.sample().incrementCount();

    return eof;
}

template <typename T>
void PositionGenerator<T>::generateSamplePeriod(const double samples_per_second)
{
    oat::Sample::Seconds period(1.0 / samples_per_second);
    sample_period_in_sec_ = period; // Auto conversion
}

// Explicit declaration to get around link errors due to this being in its own
// implementation file
template class PositionGenerator<oat::Position2D>;

} /* namespace oat */
