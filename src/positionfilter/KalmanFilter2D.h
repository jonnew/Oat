//******************************************************************************
//* File:   KalmanFilter2D.h
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

#ifndef OAT_KALMANFILTER2D_H
#define	OAT_KALMANFILTER2D_H

#include "PositionFilter.h"

#include <string>
#include <opencv2/opencv.hpp>

namespace oat {

/**
 * A 2D Kalman filter.
 */
class KalmanFilter2D : public PositionFilter {

public:

    /**
     * A 2D Kalman filter.
     * The assumed model is normally distributed constant force applied at each
     * time steps causes a random, constant acceleration in between each time-step.
     * Measurement noise is assumed to be Gaussian with a user supplied variance.
     * Model parameters (time step size, standard deviation of random acceleration,
     * and measurement noise standard deviation, etc) are supplied using the
     * configure method.
     * @param position_source_address Un-filtered position SOURCE name
     * @param position_sink_address Filtered position SINK name
     */
    KalmanFilter2D(const std::string &position_source_address,
                   const std::string& position_sink_address);

    void appendOptions(po::options_description &opts) override;
    void configure(const po::variables_map &vm) override;

private:

    // Kalman state estimate and measurement vectors
    cv::Mat_<double> kf_predicted_state_ {4, 1, CV_64F};
    cv::Mat_<double> kf_meas_ {2, 1, CV_64F};

    // Sample period
    double dt_ {0.02};

    // Standard deviation of assumed random accelerations.
    double sig_accel_ {5.0};
    double sig_measure_noise_ {0.0};

    // Parameter tuning
    std::string tuning_image_title_;
    bool tuning_windows_created_ {false};
    int sig_accel_tune_;
    int sig_measure_noise_tune_;
    bool tuning_on_ {false};
    float draw_scale_ {10.0};

    // Variables and parameters to control whether or not to apply the filter
    bool found_ {false};
    int not_found_count_ {0};
    int not_found_count_threshold_ {0};

    // Kalman filter object
    cv::KalmanFilter kf_ {4, 2, 0, CV_64F};

    /**
     * Perform Kalman filtering.
     * @param position Position to filter
     */
    void filter(oat::Position2D& position) override;

    // TODO: These subroutines have pretty boring type signatures...
    void tune(void);
    void initializeFilter(void);
    void initializeStaticMatracies(void);
    void createTuningWindows(void);
    void drawPosition(cv::Mat& canvas, const oat::Position2D& position);
};

}      /* namespace oat */
#endif /* OAT_KALMANFILTER2D_H */
