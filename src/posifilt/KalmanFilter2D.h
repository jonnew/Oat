//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
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

class KalmanFilter2D : public PositionFilter {
    
public:
    KalmanFilter2D(const std::string& position_source_name, const std::string& position_sink_name);

    bool grabPosition(void);
    void filterPosition(void);
    void serveFilteredPosition(void);

    /**
     * Configure filter parameters using a configuration file.
     * @param config_file Path to the configuration file
     * @param config_key Configuration file key specifying the table used to
     * configure the Kalman filter.
     */
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
    std::string slider_title;
    bool tuning_on;
    float draw_scale;

    // Variables and parameters to control whether or not to apply the filter
    bool found;
    int not_found_count;
    int not_found_count_threshold;

    cv::KalmanFilter kf;
    void tune(void);
    void initializeFilter(void);
    void initializeStaticMatracies(void);
    void createTuningWindows(void);
    void drawPosition(cv::Mat& canvas, const datatypes::Position2D& position);
};

#endif	/* KALMANFILTER2D_H */

