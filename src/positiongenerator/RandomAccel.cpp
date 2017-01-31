//******************************************************************************
//* File:   RandomAccel.cpp
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

#include <string>
#include <array>
#include <opencv2/opencv.hpp>

#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/datatypes/Position2D.h"

#include "RandomAccel.h"

namespace oat {

po::options_description RandomAccel::options() const
{
    // Update CLI options
    // Start with base options
    po::options_description local_opts(baseOptions());

    // Add local options
    local_opts.add_options()
        ("sigma-accel,a", po::value<double>(),
         "Standard deviation of normally-distributed random accelerations")
        ;

    return local_opts;
}

void RandomAccel::applyConfiguration(const po::variables_map &vm,
                                        const config::OptionTable &config_table)
{
    // Rate
    double fs = 1e8; // Very fast s.t. process cannot keep up
    if (oat::config::getNumericValue<double>(vm, config_table, "rate", fs, 0)) {
        enforce_sample_clock_ = true;
    } else {
        tick_ = clock_.now();
    }
    generateSamplePeriod(fs);

    // Number of samples
    oat::config::getNumericValue<uint64_t>(
        vm, config_table, "num-samples", num_samples_, 0);

    // DistanceUnit
    int d = 0;
    if (oat::config::getNumericValue<int>(
            vm, config_table, "unit-of-length", d, 0)) {
        dist_unit_ = oat::Pose::DistanceUnit(d);
    }

    // Room
    std::vector<double> r;
    if (oat::config::getArray<double, 6>(vm, config_table, "room", r)) {

        if (r[1] < 0 || r[2] < 0 || r[5] < 0) {
            throw std::runtime_error("Room width, length, and height must be "
                                     "greater than or equal to 0.");
        }
        room_.x = r[0];
        room_.width = r[1];
        room_.y = r[2];
        room_.length = r[3];
        room_.z = r[4];
        room_.height = r[5];
    }

    // Acceleration
    double a;
    if (oat::config::getNumericValue<double>(vm, config_table, "sigma-accel", a)) {
        accel_distribution_.param(
            std::normal_distribution<double>::param_type(0, a));
    }

    // Configure STM
    createStaticMatracies();
}

bool RandomAccel::generatePosition(oat::Pose &pose)
{

    if (it_ < num_samples_) {

        // Simulate one step of random, but smooth, motion
        simulateMotion();

        // Simulated pose info
        pose.found = true;
        std::array<double, 3> p{{state_(0), state_(2), state_(4)}};
        pose.set_position(p);

        it_++;

        return false;
    }

    return true;
}

void RandomAccel::simulateMotion()
{
    // Generate random acceleration
    cv::Matx31d accel_vec;
    accel_vec(0) = accel_distribution_(accel_generator_);
    accel_vec(1) = accel_distribution_(accel_generator_);
    accel_vec(2) = accel_distribution_(accel_generator_);

    // Apply acceleration and transition matrix to the simulated position
    state_ = state_transition_mat_ * state_ + input_mat_ * accel_vec;

    // Apply circular boundary (not technically correct since positive test
    // condition should result in state_(0) = 2*room_.x + room_.width - state_(0),
    // but takes care of endless oscillation that would result if
    // |state_(0) - room_.x | > room.width.
    if (state_(0) < room_.x)
        state_(0) = room_.x + room_.width;

    if (state_(0) > room_.x + room_.width)
        state_(0) = room_.x;

    if (state_(2) < room_.y)
        state_(2) = room_.y + room_.length;

    if (state_(2) > room_.y + room_.length)
        state_(2) = room_.y;

    if (state_(4) < room_.z)
        state_(4) = room_.z + room_.height;

    if (state_(4) > room_.z + room_.height)
        state_(4) = room_.z;
}

void RandomAccel::createStaticMatracies()
{
    double Ts = sample_period_in_sec_.count();

    // State transition matrix
    state_transition_mat_(0, 0) = 1.0;
    state_transition_mat_(0, 1) = Ts;
    state_transition_mat_(0, 2) = 0.0;
    state_transition_mat_(0, 3) = 0.0;
    state_transition_mat_(0, 4) = 0.0;
    state_transition_mat_(0, 5) = 0.0;

    state_transition_mat_(1, 0) = 0.0;
    state_transition_mat_(1, 1) = 1.0;
    state_transition_mat_(1, 2) = 0.0;
    state_transition_mat_(1, 3) = 0.0;
    state_transition_mat_(1, 4) = 0.0;
    state_transition_mat_(1, 5) = 0.0;

    state_transition_mat_(2, 0) = 0.0;
    state_transition_mat_(2, 1) = 0.0;
    state_transition_mat_(2, 2) = 1.0;
    state_transition_mat_(2, 3) = Ts;
    state_transition_mat_(2, 4) = 0.0;
    state_transition_mat_(2, 5) = 0.0;

    state_transition_mat_(3, 0) = 0.0;
    state_transition_mat_(3, 1) = 0.0;
    state_transition_mat_(3, 2) = 0.0;
    state_transition_mat_(3, 3) = 1.0;
    state_transition_mat_(3, 4) = 0.0;
    state_transition_mat_(3, 5) = 0.0;

    state_transition_mat_(4, 0) = 0.0;
    state_transition_mat_(4, 1) = 0.0;
    state_transition_mat_(4, 2) = 0.0;
    state_transition_mat_(4, 3) = 0.0;
    state_transition_mat_(4, 4) = 1.0;
    state_transition_mat_(4, 5) = Ts;

    state_transition_mat_(5, 0) = 0.0;
    state_transition_mat_(5, 1) = 0.0;
    state_transition_mat_(5, 2) = 0.0;
    state_transition_mat_(5, 3) = 0.0;
    state_transition_mat_(5, 4) = 0.0;
    state_transition_mat_(5, 5) = 1.0;

    // Input Matrix
    input_mat_(0, 0) = (Ts*Ts)/2.0;
    input_mat_(0, 1) = 0.0;
    input_mat_(0, 2) = 0.0;

    input_mat_(1, 0) = Ts;
    input_mat_(1, 1) = 0.0;
    input_mat_(1, 2) = 0.0;

    input_mat_(2, 0) = 0.0;
    input_mat_(2, 1) = (Ts*Ts)/2.0;
    input_mat_(2, 2) = 0.0;

    input_mat_(3, 0) = 0.0;
    input_mat_(3, 1) = Ts;
    input_mat_(3, 2) = 0.0;

    input_mat_(4, 0) = 0.0;
    input_mat_(4, 1) = 0.0;
    input_mat_(4, 2) = (Ts*Ts)/2.0;

    input_mat_(5, 0) = 0.0;
    input_mat_(5, 1) = 0.0;
    input_mat_(5, 2) = Ts;
}

} /* namespace oat */
