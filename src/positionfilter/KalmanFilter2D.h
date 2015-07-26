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

#ifndef KALMANFILTER2D_H
#define	KALMANFILTER2D_H

#include <string>
#include <opencv2/opencv.hpp>

#include "PositionFilter.h"

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
     * @param position_source_name Un-filtered position SOURCE name
     * @param position_sink_name Filtered position SINK name
     */
    KalmanFilter2D(const std::string& position_source_name, const std::string& position_sink_name);

    void configure(const std::string& config_file, const std::string& config_key);

private:
    
    // Kalman state estimate and measurement vectors
    cv::Mat kf_predicted_state;
    cv::Mat kf_meas;

    // Sample period
    double dt;

    // Standard deviation of assumed random accelerations.
    double sig_accel;
    double sig_measure_noise;

    // Parameter tuning
    bool tuning_windows_created;
    int sig_accel_tune;
    int sig_measure_noise_tune;
    bool tuning_on;
    float draw_scale;

    // Variables and parameters to control whether or not to apply the filter
    bool found;
    int not_found_count;
    int not_found_count_threshold;
    
    // Filtered position
    oat::Position2D filtered_position;

    // Tuning window name
    std::string tuning_image_title;

    // Kalman filter object
    cv::KalmanFilter kf;
    
    /**
     * Perform Kalman filtering.
     * @param position_in Un-filtered position SOURCE
     * @return filtered position
     */
    oat::Position2D filterPosition(oat::Position2D& position_in);
    
    // TODO: These subroutines have pretty boring type signatures...
    void tune(void);
    void initializeFilter(void);
    void initializeStaticMatracies(void);
    void createTuningWindows(void);
    void drawPosition(cv::Mat& canvas, const oat::Position2D& position);
};

#endif	/* KALMANFILTER2D_H */

