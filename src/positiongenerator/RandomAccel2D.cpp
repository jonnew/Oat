//******************************************************************************
//* File:   RandomAccel2D.cpp
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
//*****************************************************************************

#include <iostream>
#include <math.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <cpptoml.h>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/OatTOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

#include "RandomAccel2D.h"

namespace oat {

RandomAccel2D::RandomAccel2D(const std::string &position_sink_address,
                             const double samples_per_second,
                             const int64_t num_samples) :
  PositionGenerator<oat::Position2D>(position_sink_address, samples_per_second, num_samples)
{
    createStaticMatracies();
}

void RandomAccel2D::configure(const std::string &config_file,
                              const std::string &config_key) {

    // Available options
    std::vector<std::string> options {"dt",
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
        double dt; 
        if (oat::config::getValue(this_config, "dt", dt, 0)) {
             generateSamplePeriod(1.0/dt);
        }

        // Number of position samples
        oat::config::getValue(this_config, "num-samples", num_samples_, 0); // TODO: This should be part of the base class

        // Camera Matrix
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

bool RandomAccel2D::generatePosition(oat::Position2D &position) {

    if (it_ < num_samples_) { // TODO: This should be part of the base class

        // Simulate one step of random, but smooth, motion
        simulateMotion();

        // Simulated position info
        position.position_valid = true;
        position.position.x = state_(0);
        position.position.y = state_(2);

        // We have access to the velocity info for comparison
        position.velocity_valid = true;
        position.velocity.x = state_(1);
        position.velocity.y = state_(3);

        it_++;

        return false;
    }

    return true;
}

void RandomAccel2D::simulateMotion() {

    // Generate random acceleration
    accel_vec_(0) = accel_distribution_(accel_generator_);
    accel_vec_(1) = accel_distribution_(accel_generator_);

    // Apply acceleration and transition matrix to the simulated position
    state_ = state_transition_mat_ * state_ + input_mat_ * accel_vec_;

    // Apply circular boundary (not technically correct since positive test
    // condition should result in state_(0) = 2*room_.x + room_.width - state_(0),
    // but takes care of endless oscillation that would result if
    // |state_(0) - room_.x | > room.width.
    if (state_(0) < room_.x)
        state_(0) = room_.x + room_.width;

    if (state_(0) > room_.x + room_.width)
        state_(0) = room_.x;

    if (state_(2) < room_.y)
        state_(2) = room_.y + room_.height;

    if (state_(2) > room_.y + room_.height)
        state_(2) = room_.y;
}

void RandomAccel2D::createStaticMatracies() {

    double Ts = sample_period_in_sec_.count();

    // State transition matrix
    state_transition_mat_(0, 0) = 1.0;
    state_transition_mat_(0, 1) = Ts;
    state_transition_mat_(0, 2) = 0.0;
    state_transition_mat_(0, 3) = 0.0;

    state_transition_mat_(1, 0) = 0.0;
    state_transition_mat_(1, 1) = 1.0;
    state_transition_mat_(1, 2) = 0.0;
    state_transition_mat_(1, 3) = 0.0;

    state_transition_mat_(2, 0) = 0.0;
    state_transition_mat_(2, 1) = 0.0;
    state_transition_mat_(2, 2) = 1.0;
    state_transition_mat_(2, 3) = Ts;

    state_transition_mat_(3, 0) = 0.0;
    state_transition_mat_(3, 1) = 0.0;
    state_transition_mat_(3, 2) = 0.0;
    state_transition_mat_(3, 3) = 1.0;

    // Input Matrix
    input_mat_(0, 0) = (Ts*Ts)/2.0;
    input_mat_(0, 1) = 0.0;

    input_mat_(1, 0) = Ts;
    input_mat_(1, 1) = 0.0;

    input_mat_(2, 0) = 0.0;
    input_mat_(2, 1) = (Ts*Ts)/2.0;

    input_mat_(3, 0) = 0.0;
    input_mat_(3, 1) = Ts;

}

} /* namespace oat */
