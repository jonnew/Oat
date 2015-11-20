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
                                     const double samples_per_second) :
  name_("testpos[*->" + position_sink_address + "]")
, position_sink_address_(position_sink_address)
{

  generateSamplePeriod(samples_per_second);
  tick = clock.now();
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
    generatePosition(internal_position_);

    // This is pure SINK, so it increments the sample count
    internal_position_.incrementSampleCount();

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    position_sink_.wait();

    *shared_position_ = internal_position_;

    // Tell sources there is new data
    position_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // This sink never reaches END state
    return false;
}

template<typename T>
void PositionGenerator<T>::generateSamplePeriod(const double samples_per_second) {

    std::chrono::duration<double> period {1.0 / samples_per_second};

    // Automatic conversion
    sample_period_in_sec_ = period;
    internal_position_.set_sample_period_sec(period.count());
}

// Explicit declaration to get around link errors due to this being in its own
// implementation file
template class PositionGenerator<oat::Position2D>;

} /* namespace oat */