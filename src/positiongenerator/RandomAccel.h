//******************************************************************************
//* File:   RandomAccel.h
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

#ifndef OAT_RANDOMACCEL_H
#define	OAT_RANDOMACCEL_H

#include <random>
#include <string>

#include <opencv2/core/mat.hpp>

#include "PoseGenerator.h"

namespace oat {

class RandomAccel : public PoseGenerator {

public:
    /**
     * A 3D Gaussian random acceleration generator.
     * Test poses are subject to random, uncorrelated 3D, Gaussian
     * accelerations. 
     */
    using PoseGenerator::PoseGenerator;

private:
    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    // Random number generator and distributions
    std::default_random_engine accel_gen_{std::random_device{}()};
    std::normal_distribution<double> pos_accel_dist_{
        0.0, default_side_ / 20.0}; // pose.unit_of_length/sec^2
    std::normal_distribution<double> orient_accel_dist_{0.0, 5.0}; // deg/sec^2

    // Simulated pose (3DOF position, 3DOF orientation)
    cv::Matx<double, 12, 1> state_{
        default_side_ / 2.0, // x
        0.0,                 // x'
        default_side_ / 2.0, // y
        0.0,                 // y'
        default_side_ / 2.0, // z
        0.0,                 // z'
        0.0,                 // psi
        0.0,                 // psi'
        0.0,                 // theta
        0.0,                 // theta'
        0.0,                 // phi
        0.0,                 // phi'
    };

    // STM and input matrix
    cv::Matx<double, 12, 12> state_transition_mat_;
    cv::Matx<double, 12, 6> input_mat_;

    bool produce_orientation_{false};

    bool generatePosition(oat::Pose &position) override;
    void createStaticMatracies(void);
    void simulateMotion(void);
};

}      /* namespace oat */
#endif /* OAT_RANDOMACCEL_H */
