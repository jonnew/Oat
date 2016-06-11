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
    } else {

        // Need something so that positions are not nonsense
        generateSamplePeriod(10000.0);
    }

    tick_ = clock_.now();
}

template <typename T>
PositionGenerator<T>::~PositionGenerator() { }

template <typename T>
void PositionGenerator<T>::configure(const std::string &config_file,
                                     const std::string &config_key) {

    // Available options
    std::vector<std::string> options {"rate-hz",
                                      "num-samples",
                                      "room"};

    // This will throw cpptoml::parse_exception if a file
    // with invalid TOML is provided
    auto config = cpptoml::parse_file(config_file);

    // See if a camera configuration was provided
    if (config->contains(config_key)) {

        // Get this components configuration table
        auto this_config = config->get_table(config_key);

        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);

        // Sample generation period
        double rate_hz;
        if (oat::config::getValue(this_config, "rate-hz", rate_hz, 0))
             generateSamplePeriod(rate_hz);

        // Number of position samples
        oat::config::getValue(this_config, "num-samples", num_samples_, 0);

        // Periodic boundaries for the generated positions
        oat::config::Array room_array;
        if (oat::config::getArray(this_config, "room", room_array, 4, false)) {

            auto room_vec = room_array->array_of<double>();
            room_.x      = room_vec[0]->get();
            room_.y      = room_vec[1]->get();
            room_.width  = room_vec[2]->get();
            room_.height = room_vec[3]->get();
        }

    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

template<typename T>
void PositionGenerator<T>::connectToNode() {

    // Bind to sink sink node and create a shared position
    position_sink_.bind(position_sink_address_, position_sink_address_);
    shared_position_ = position_sink_.retrieve();
    shared_position_->sample().set_rate_hz(1.0 / sample_period_in_sec_.count());
}

template<typename T>
bool PositionGenerator<T>::process() {

    // Generate internal position
    bool eof = generatePosition(internal_position_);

    if (enforce_sample_clock_) {
        auto tock = clock_.now();
        std::this_thread::sleep_for(sample_period_in_sec_ - (tock - tick_));
        tick_ = clock_.now();
    }

    // This is a pure SINK so it increments the sample count
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

    oat::Sample::Seconds period(1.0 / samples_per_second);

    // Automatic conversion
    sample_period_in_sec_ = period;
}

// Explicit declaration to get around link errors due to this being in its own
// implementation file
template class PositionGenerator<oat::Position2D>;

} /* namespace oat */
