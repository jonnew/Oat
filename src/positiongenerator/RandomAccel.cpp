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

#include <algorithm>
#include <array>
#include <string>

#include <opencv2/opencv.hpp>

#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/datatypes/Pose.h"

#include "RandomAccel.h"

namespace oat {

po::options_description RandomAccel::options() const
{
    // Update CLI options
    // Start with base options
    po::options_description local_opts(baseOptions());

    // Add local options
    local_opts.add_options()
        ("position-sigma-accel,a", po::value<double>(),
         "Standard deviation of normally-distributed random positional "
         "accelerations. Specified unit-of-length/sec^2.")
        ("orientation-sigma-accel,b", po::value<double>(),
         "Standard deviation of normally-distributed random orientation "
         "accelerations. Specified in degrees/sec^2.")
        ;

    return local_opts;
}

void RandomAccel::applyConfiguration(const po::variables_map &vm,
                                     const config::OptionTable &config_table)
{
    // Rate
    double fs = 1e8; // Very fast s.t. process cannot keep up
    if (oat::config::getNumericValue<double>(vm, config_table, "rate", fs, 0))
        enforce_sample_clock_ = true;
    else
        tick_ = clock_.now();

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
    if (oat::config::getNumericValue<double>(
            vm, config_table, "position-sigma-accel-accel", a)) {
        pos_accel_dist_.param(
            std::normal_distribution<double>::param_type(0, a));
    }

    if (oat::config::getNumericValue<double>(
            vm, config_table, "orientation-sigma-accel-accel", a)) {
        pos_accel_dist_.param(
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
        pose.position_dof = Pose::DOF::Three;
        pose.orientation_dof = Pose::DOF::Three;
        std::array<double, 3> p{{state_(0), state_(2), state_(4)}};
        pose.set_position(p);
        std::array<double, 3> o{{state_(6), state_(8), state_(10)}};
        pose.fromTaitBryan(o, true);

        it_++;

        return false;
    }

    return true;
}

void RandomAccel::simulateMotion()
{
    // Generate random acceleration
    cv::Matx61d accel_vec;
    accel_vec(0) = pos_accel_dist_(accel_gen_);
    accel_vec(1) = pos_accel_dist_(accel_gen_);
    accel_vec(2) = pos_accel_dist_(accel_gen_);
    accel_vec(3) = orient_accel_dist_(accel_gen_);
    accel_vec(4) = orient_accel_dist_(accel_gen_);
    accel_vec(5) = orient_accel_dist_(accel_gen_);

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

    // Zero out the the STM and input matricies
    std::fill(std::begin(state_transition_mat_.val),
              std::end(state_transition_mat_.val),
              0);
    std::fill(std::begin(input_mat_.val), std::end(input_mat_.val), 0);

    // STM is:
    // [A 0]
    // [0 A]
    // Where A is:
    // [1  Ts 0  0  0  0 ]
    // [0  1  0  0  0  0 ]
    // [0  0  1  Ts 0  0 ]
    // [0  0  0  1  0  0 ]
    // [0  0  0  0  1  Ts]
    // [0  0  0  0  0  1 ]
    state_transition_mat_(0, 0)   = 1.0;
    state_transition_mat_(0, 1)   = Ts;
    state_transition_mat_(1, 1)   = 1.0;
    state_transition_mat_(2, 2)   = 1.0;
    state_transition_mat_(2, 3)   = Ts;
    state_transition_mat_(3, 3)   = 1.0;
    state_transition_mat_(4, 4)   = 1.0;
    state_transition_mat_(4, 5)   = Ts;
    state_transition_mat_(5, 5)   = 1.0;
    state_transition_mat_(6, 6)   = 1.0;
    state_transition_mat_(6, 7)   = Ts;
    state_transition_mat_(7, 7)   = 1.0;
    state_transition_mat_(8, 8)   = 1.0;
    state_transition_mat_(8, 9)   = Ts;
    state_transition_mat_(9, 9)   = 1.0;
    state_transition_mat_(10, 10) = 1.0;
    state_transition_mat_(10, 11) = Ts;
    state_transition_mat_(11, 11) = 1.0;

    // Input Matrix:
    // [U 0]
    // [0 U]
    // Where U is:
    // [Ts^2/2 0       0     ]
    // [Ts     0       0     ]
    // [0      Ts^2/2  0     ]
    // [0      Ts      0     ]
    // [0      0       Ts^2/2]
    // [0      0       Ts    ]
    input_mat_(0, 0)  = (Ts*Ts)/2.0;
    input_mat_(1, 0)  = Ts;
    input_mat_(2, 1)  = (Ts*Ts)/2.0;
    input_mat_(3, 1)  = Ts;
    input_mat_(4, 2)  = (Ts*Ts)/2.0;
    input_mat_(5, 2)  = Ts;
    input_mat_(6, 3)  = (Ts*Ts)/2.0;
    input_mat_(7, 3)  = Ts;
    input_mat_(8, 4)  = (Ts*Ts)/2.0;
    input_mat_(9, 4)  = Ts;
    input_mat_(10, 5) = (Ts*Ts)/2.0;
    input_mat_(11, 5) = Ts;
}

} /* namespace oat */
