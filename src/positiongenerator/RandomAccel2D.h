//******************************************************************************
//* File:   RandomAccel2D.h
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

#ifndef OAT_RANDOMACCEL2D_H
#define	OAT_RANDOMACCEL2D_H

#include <chrono>
#include <random>
#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/datatypes/Position2D.h"

#include "PositionGenerator.h"

namespace oat {

/**
 * A 2D Gaussian random acceleration position generator.
 */
class RandomAccel2D : public PositionGenerator<oat::Position2D> {

public:

    /**
     * A 2D Gaussian random acceleration generator.
     * Test positions are subject to random, uncorrelated 2D, Gaussian
     * accelerations.
     */
    RandomAccel2D(const std::string &position_sink_address,
                  const double samples_per_second,
                  const int64_t num_samples);

    void configure(const std::string &file_name,
                   const std::string &key) override;

private:

    // Random number generator
    std::default_random_engine accel_generator_ {std::random_device{}()};
    std::normal_distribution<double> accel_distribution_ {0.0, 100.0};

    // Simulated position
    cv::Matx41d state_ {0.0, 0.0, 0.0, 0.0}; // Should be center of bounding region
    cv::Matx21d accel_vec_;

    // STM and input matrix
    cv::Rect_<double> room_ {0, 0, 728, 728}; //!< "Room" circular boundaries in which simulated particle resides.
    cv::Matx44d state_transition_mat_;
    cv::Matx<double, 4, 2> input_mat_;

    bool generatePosition(oat::Position2D &position) override;
    void createStaticMatracies(void);
    void simulateMotion(void);

};

}      /* namespace oat */
#endif /* OAT_RANDOMACCEL2D_H */

