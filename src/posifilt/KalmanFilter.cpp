//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.

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

#include "KalmanFilter.h"
#include "../../lib/cpptoml/cpptoml.h"

/**
 * A Kalman filter for position measures. The assumed model is normally distributed
 * constant force applied at each time steps causes a random, constant accelleration
 * in between each time-step.
 * 
 * @param position_source_name The position measure SOURCE
 * @param position_sink_name The filtered position SINK
 */
KalmanFilter::KalmanFilter(std::string position_source_name, std::string position_sink_name) :
PositionFilter(position_source_name, position_sink_name)
, dt(0.03333333333333333)
, kf(6, 3, 0)
, kf_state(6, 1, CV_32F)
, kf_meas(3, 1, CV_32F)
, found(false)
, not_found_count_threshold(100) // TODO: good idea?
, sig_accel(0.1)
, sig_measure_noise(0.01)
, tuning_windows_created(false)
, slider_title(position_sink_name + "_sliders") {
}

void KalmanFilter::grabPosition() {

    position_source.get_value(raw_position);

    // Transform raw position into kf_meas vector
    if (raw_position.position_valid) {
        kf_meas.at<float>(0) = raw_position.position.x;
        kf_meas.at<float>(1) = raw_position.position.y;
        kf_meas.at<float>(2) = raw_position.position.z;
        not_found_count = 0;

        // We are coming from a time step where there were no measurements for a
        // long time, so we need to reinitialize the filter
        if (!found) {
            initializeFilter();
        }

        found = true;
    }
    else {
        not_found_count++;
    }
}

void KalmanFilter::filterPosition() {

    // Tune the filter, if requested
    tune();
    
    // If we have not gotten a measurement of the object for a long time
    // we need to reinitialize the filter
    if (not_found_count >= not_found_count_threshold) {
        found = false;
    }
    else {
        kf.statePost = kf_state;
    }

    // Apply the Kalman update
    kf.correct(kf_meas);

    // Produce state estimate
    kf_state = kf.predict();
}

void KalmanFilter::serveFilteredPosition() {

    // Create a new Position object from the kf_state
    shmem::Position filtered_position;
    filtered_position.position.x = kf_state.at<float>(0);
    filtered_position.velocity.x = kf_state.at<float>(1);
    filtered_position.position.x = kf_state.at<float>(2);
    filtered_position.velocity.y = kf_state.at<float>(3);
    filtered_position.position.x = kf_state.at<float>(4);
    filtered_position.velocity.z = kf_state.at<float>(5);

    // This Position is only valid if the not_found_count_threshold has not
    // be exceeded
    if (found) {
        filtered_position.position_valid = true;
        filtered_position.velocity_valid = true;
    } else {
        filtered_position.position_valid = false;
        filtered_position.velocity_valid = false;
    }
}

void KalmanFilter::configure(std::string config_file, std::string config_key) {
    
    cpptoml::table config;

    try {
        config = cpptoml::parse_file(config_file);
    } catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse " << config_file << ": " << e.what() << std::endl;
    }

    try {
        // See if a camera configuration was provided
        if (config.contains(config_key)) {

            auto this_config = *config.get_table(config_key);

            if (this_config.contains("dt")) {
                dt = (float)(*this_config.get_as<double>("dt"));
            }
            
            if (this_config.contains("not_found_timeout")) {
                // Seconds to samples
                float timeout_in_sec = (float)(*this_config.get_as<double>("not_found_timeout"));
                not_found_count_threshold = (int)(timeout_in_sec/dt);
            }
            
            if (this_config.contains("sigma_accel")) {
                sig_accel = (float)(*this_config.get_as<double>("sigma_accel"));
            }
            
            if (this_config.contains("sigma_noise")) {
                sig_measure_noise = (float)(*this_config.get_as<double>("sigma_noise"));
            }
            
            if (this_config.contains("tune")) {
                if (*this_config.get_as<bool>("tune")) {
                    tuning_on = true;
                    createTuningWindows();
                } 
            }

        } else {
            std::cerr << "No Kalman Filter configuration named \"" + config_key + "\" was provided. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void KalmanFilter::initializeFilter(void) {

    // State transition matrix
    // [ 1  dt 0  0  0  0 ]
    // [ 0  1  0  0  0  0 ]
    // [ 0  0  1  dt 0  0 ]
    // [ 0  0  0  1  0  0 ]
    // [ 0  0  0  0  1  dt]
    // [ 0  0  0  0  0  1 ]    
    cv::setIdentity(kf.transitionMatrix);
    kf.transitionMatrix.at<float>(1, 2) = dt;
    kf.transitionMatrix.at<float>(3, 4) = dt;
    kf.transitionMatrix.at<float>(5, 6) = dt;


    // Observation Matrix (can only see position directly)
    // [ 1  0  0  0  0  0 ]
    // [ 0  0  1  0  0  0 ]
    // [ 0  0  0  0  1  0 ]
    cv::Mat::zeros(3, 6, CV_32F);
    kf.measurementMatrix.at<float>(1, 1) = 1.0;
    kf.measurementMatrix.at<float>(2, 3) = 1.0;
    kf.measurementMatrix.at<float>(3, 5) = 1.0;

    // Noise covariance matrix (see pp13-15 of MWL.JPN.105.02.002 for derivation)
    // [ dt^4/2 dt^2/3 				   ] 
    // [ dt^2/3 dt^2   				   ]
    // [               dt^4/2 dt^2/3 		   ]
    // [               dt^2/3 dt^2  		   ] * sigma_accel^2
    // [                             dt^4/2 dt^2/3 ]
    // [                             dt^2/3 dt^2   ]
    kf.processNoiseCov.at<float>(1, 1) = sig_accel * sig_accel * (dt * dt * dt * dt) / 4.0;
    kf.processNoiseCov.at<float>(1, 2) = sig_accel * sig_accel * (dt * dt * dt) / 2.0;
    kf.processNoiseCov.at<float>(1, 3) = 0.0;
    kf.processNoiseCov.at<float>(1, 4) = 0.0;
    kf.processNoiseCov.at<float>(1, 5) = 0.0;
    kf.processNoiseCov.at<float>(1, 6) = 0.0;
    kf.processNoiseCov.at<float>(2, 1) = sig_accel * sig_accel * (dt * dt * dt) / 2.0;
    kf.processNoiseCov.at<float>(2, 2) = sig_accel * sig_accel * (dt * dt);
    kf.processNoiseCov.at<float>(2, 3) = 0.0;
    kf.processNoiseCov.at<float>(2, 4) = 0.0;
    kf.processNoiseCov.at<float>(2, 5) = 0.0;
    kf.processNoiseCov.at<float>(2, 6) = 0.0;
    kf.processNoiseCov.at<float>(3, 1) = 0.0;
    kf.processNoiseCov.at<float>(3, 2) = 0.0;
    kf.processNoiseCov.at<float>(3, 3) = sig_accel * sig_accel * (dt * dt * dt * dt) / 4.0;
    kf.processNoiseCov.at<float>(3, 4) = sig_accel * sig_accel * (dt * dt * dt) / 2.0;
    kf.processNoiseCov.at<float>(3, 5) = 0.0;
    kf.processNoiseCov.at<float>(3, 6) = 0.0;
    kf.processNoiseCov.at<float>(4, 1) = 0.0;
    kf.processNoiseCov.at<float>(4, 2) = 0.0;
    kf.processNoiseCov.at<float>(4, 3) = sig_accel * sig_accel * (dt * dt * dt) / 2.0;
    kf.processNoiseCov.at<float>(4, 4) = sig_accel * sig_accel * (dt * dt);
    kf.processNoiseCov.at<float>(4, 5) = 0.0;
    kf.processNoiseCov.at<float>(4, 6) = 0.0;
    kf.processNoiseCov.at<float>(5, 1) = 0.0;
    kf.processNoiseCov.at<float>(5, 2) = 0.0;
    kf.processNoiseCov.at<float>(5, 3) = 0.0;
    kf.processNoiseCov.at<float>(5, 4) = 0.0;
    kf.processNoiseCov.at<float>(5, 5) = sig_accel * sig_accel * (dt * dt * dt * dt) / 4.0;
    kf.processNoiseCov.at<float>(5, 6) = sig_accel * sig_accel * (dt * dt * dt) / 2.0;
    kf.processNoiseCov.at<float>(6, 1) = 0.0;
    kf.processNoiseCov.at<float>(6, 2) = 0.0;
    kf.processNoiseCov.at<float>(6, 3) = 0.0;
    kf.processNoiseCov.at<float>(6, 4) = 0.0;
    kf.processNoiseCov.at<float>(6, 5) = sig_accel * sig_accel * (dt * dt * dt) / 2.0;
    kf.processNoiseCov.at<float>(6, 6) = sig_accel * sig_accel * (dt * dt);

    // Measurement noise covariance
    // [ sig_x^2  0  0 ]
    // [ 0  sig_y^2  0 ]
    // [ 0  0  sig_z^2 ]
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(sig_measure_noise * sig_measure_noise));

    // Error covariance matrix (initialize with large value to indicate a lack
    // of trust in the model)
    cv::setIdentity(kf.errorCovPre, 10.0);

    // TODO: Add head direction?
    // The state is
    // [ x  x'  y  y'  z  z' ], where ' denotes the time derivative
    // Initialize the state using the current measurement
    kf_state.at<float>(0) = kf_meas.at<float>(0);
    kf_state.at<float>(1) = 0.0;
    kf_state.at<float>(2) = kf_meas.at<float>(1);
    kf_state.at<float>(3) = 0.0;
    kf_state.at<float>(4) = kf_meas.at<float>(2);
    kf_state.at<float>(5) = 0.0;
}

void KalmanFilter::tune() {

    if (tuning_on) {
        if (!tuning_windows_created) {
            createTuningWindows();
        }
        cv::waitKey(1);
    } else if (!tuning_on && tuning_windows_created) {
        // Destroy the tuning windows
        cv::destroyWindow(slider_title);
        tuning_windows_created = false;
    }
}

void KalmanFilter::createTuningWindows() {
    
    // Create window for sliders
    cv::namedWindow(slider_title, cv::WINDOW_AUTOSIZE);

    // Create sliders and insert them into window
    cv::createTrackbar("SIGMA ACCEL.", slider_title, &sig_accel, 10); 
    cv::createTrackbar("SIGMA NOISE", slider_title, &sig_measure_noise, 10); 
    
    tuning_windows_created = true;
}
