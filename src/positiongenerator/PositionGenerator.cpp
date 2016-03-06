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

#include "PositionGenerator.h"

namespace oat {

template<typename T>
PositionGenerator<T>::PositionGenerator(const std::string &position_sink_address,
                                        const double samples_per_second, 
                                        const int64_t num_samples) :
  num_samples_(num_samples - 1)
, name_("posigen[*->" + position_sink_address + "]")
, position_sink_address_(position_sink_address)
{
    if (samples_per_second > 0) {
        enforce_sample_clock_ = true;
        generateSamplePeriod(samples_per_second);
    }

    tick_ = clock_.now();
}

template<typename T>
void PositionGenerator<T>::connectToNode() {

    // Bind to sink sink node and create a shared position
    position_sink_.bind(position_sink_address_, position_sink_address_);
    shared_position_ = position_sink_.retrieve();
}

template<typename T>
bool PositionGenerator<T>::process() {

    // Generate internal frame
    bool eof = generatePosition(internal_position_);

    // This is pure SINK, so it increments the sample count
    internal_position_.sample().incrementCount();

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    position_sink_.wait();

    *shared_position_ = internal_position_;

    // Tell sources there is new data
    position_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    return eof;
}

template<typename T>
void PositionGenerator<T>::generateSamplePeriod(const double samples_per_second) {

    std::chrono::duration<double> period {1.0 / samples_per_second};

    // Automatic conversion
    sample_period_in_sec_ = period;
    internal_position_.sample().set_period_sec(period.count());
}

// Explicit declaration to get around link errors due to this being in its own
// implementation file
template class PositionGenerator<oat::Position2D>;

} /* namespace oat */
