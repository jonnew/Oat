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

#include <string>
#include <opencv2/opencv.hpp>

#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/datatypes/Position2D.h"

#include "RandomAccel2D.h"

namespace oat {

RandomAccel2D::RandomAccel2D(const std::string &position_sink_address)
: PositionGenerator(position_sink_address)
{
    // Nothing
}

void RandomAccel2D::appendOptions(po::options_description &opts)
{
    // Accepts a config file and common opts
    PositionGenerator::appendOptions(opts);

    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("sigma-accel,a", po::value<double>(),
         "Standard deviation of normally-distributed random accelerations")
        ;
     
    opts.add(local_opts);

    // Return valid keys
    for (auto &o : local_opts.options())
        config_keys_.push_back(o->long_name());
}

void RandomAccel2D::configure(const po::variables_map &vm)
{
    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);
    
    // Rate
    double fs = 1e8; // Very fast s.t. process cannot keep up
    if (oat::config::getNumericValue<double>(
                vm, config_table, "rate", fs, 0)) {
        enforce_sample_clock_ = true;
    } else {
        tick_ = clock_.now();
    }
    generateSamplePeriod(fs);

    // Number of samples
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

    // Acceleration
    double a;
    if (oat::config::getNumericValue<double>(vm, config_table, "sigma-accel", a)) {
        accel_distribution_.param(
            std::normal_distribution<double>::param_type(0, a));
    }

    // Configure STM
    createStaticMatracies();
}

bool RandomAccel2D::generatePosition(oat::Position2D &position)
{

    if (it_ < num_samples_) {

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

void RandomAccel2D::simulateMotion() 
{
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

void RandomAccel2D::createStaticMatracies() 
{
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
