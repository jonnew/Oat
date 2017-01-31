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

#include <chrono>
#include <random>
#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/datatypes/Position2D.h"

#include "PoseGenerator.h"

namespace oat {

class RandomAccel : public PoseGenerator {

public:
    /**
     * A 3D Gaussian random acceleration generator.
     * Test positions are subject to random, uncorrelated 3D, Gaussian
     * accelerations. Test orientations are subject to uniform random
     * rotations.
     */
    using PoseGenerator::PoseGenerator;

private:
    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    // Random number generator
    std::default_random_engine accel_generator_{std::random_device{}()};
    std::normal_distribution<double> accel_distribution_{0.0, default_side_ / 20.0};

    // Simulated position
    cv::Matx61d state_{default_side_ / 2.0,
                       0.0,
                       default_side_ / 2.0,
                       0.0,
                       default_side_ / 2.0,
                       0.0};

    // STM and input matrix
    cv::Matx66d state_transition_mat_;
    cv::Matx<double, 6, 3> input_mat_;

    bool generatePosition(oat::Pose &position) override;
    void createStaticMatracies(void);
    void simulateMotion(void);
};

}      /* namespace oat */
#endif /* OAT_RANDOMACCEL_H */
