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

#ifndef KALMANFILTER_H
#define	KALMANFILTER_H

#include <string>
#include <opencv2/opencv.hpp>

#include "PositionFilter.h"

class KalmanFilter : public PositionFilter {
public:
    KalmanFilter(std::string position_source_name, std::string position_sink_name);

    void grabPosition(void);
    void filterPosition(void);
    void serveFilteredPosition(void);

    /**
     * Configure filter parameters using a configuration file.
     * @param config_file Path to the configuration file
     * @param config_key Configuration file key specifying the table used to
     * configure the Kalman filter.
     */
    void configure(std::string config_file, std::string config_key);

private:

    // Raw position from source
    shmem::Position raw_position;

    // Kalman state estimate and measurement vectors
    cv::Mat kf_state, kf_meas;

    // Sample period
    float dt;

    // Standard deviation of assumed random accelerations.
    float sig_accel;
    float sig_measure_noise;

    // Parameter tuning
    bool tuning_windows_created;
    std::string slider_title;
    bool tuning_on;

    // Variables and parameters to control whether or not to apply the filter
    bool found;
    int not_found_count;
    int not_found_count_threshold;

    cv::KalmanFilter kf;
    void tune(void);
    void initializeFilter(void);
    void createTuningWindows(void);
};

#endif	/* KALMANFILTER_H */

