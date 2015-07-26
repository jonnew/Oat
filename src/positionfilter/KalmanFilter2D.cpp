//******************************************************************************
//* File:   KalmanFilter2D.cpp
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

#include "KalmanFilter2D.h"
#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/cpptoml/OatTOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

/**
 * A Kalman filter for position measures. The assumed model is normally distributed
 * constant force applied at each time steps causes a random, constant acceleration
 * in between each time-step.
 * 
 * @param position_source_name The position measure SOURCE
 * @param position_sink_name The filtered position SINK
 */
KalmanFilter2D::KalmanFilter2D(const std::string& position_source_name, const std::string& position_sink_name) :
  PositionFilter(position_source_name, position_sink_name)
, dt(0.02)
, kf(4, 2, 0, CV_64F)
, kf_predicted_state(4, 1, CV_64F)
, kf_meas(2, 1, CV_64F)
, found(false)
, not_found_count_threshold(0)
, sig_accel(5.0)
, sig_measure_noise(0.0)
, tuning_windows_created(false)
, draw_scale(10.0)
, tuning_image_title(position_sink_name + "_tuning") {

    filtered_position.set_label(position_sink_name);
    sig_accel_tune = (int) (sig_measure_noise);
    sig_measure_noise_tune = (int) (sig_measure_noise);
}

oat::Position2D KalmanFilter2D::filterPosition(oat::Position2D& raw_position) {

    // Transform raw position into kf_meas vector
    if (raw_position.position_valid) {
        kf_meas.at<double>(0) = raw_position.position.x;
        kf_meas.at<double>(1) = raw_position.position.y;
        not_found_count = 0;

        // We are coming from a time step where there were no measurements for a
        // long time, so we need to reinitialize the filter
        if (!found) {
            initializeFilter();
        }

        found = true;
    } else {
        not_found_count++;
    }

    // If we have not gotten a measurement of the object for a long time
    // we need to reinitialize the filter
    if (not_found_count >= not_found_count_threshold) {
        found = false;
    } 
    
    // Only update if the object is found (this includes time points for which
    // the position measurement was invalid, but we are within the not_found_count_threshold)
    if (found) {
        
        kf_predicted_state = kf.predict();

        // Apply the Kalman update
        kf.correct(kf_meas);
    }
    
    filtered_position.position.x = kf_predicted_state.at<double>(0);
    filtered_position.velocity.x = kf_predicted_state.at<double>(1);
    filtered_position.position.y = kf_predicted_state.at<double>(2);
    filtered_position.velocity.y = kf_predicted_state.at<double>(3);

    // This Position is only valid if the not_found_count_threshold has not
    // be exceeded
    if (found) {
        filtered_position.position_valid = true;
        filtered_position.velocity_valid = true;
    } else {
        filtered_position.position_valid = false;
        filtered_position.velocity_valid = false;
    }
    
    // Tune the filter, if requested
    tune();
    
    return filtered_position;
}

void KalmanFilter2D::configure(const std::string& config_file, const std::string& config_key) {

    // Available options
    std::vector<std::string> options {
                "dt", 
                "timeout", 
                "sigma_accel", 
                "sigma_noise", 
                "tune" };
    
    // This will throw cpptoml::parse_exception if a file 
    // with invalid TOML is provided
    cpptoml::table config;
    config = cpptoml::parse_file(config_file);

    // See if a camera configuration was provided
    if (config.contains(config_key)) {
        
        // Get this components configuration table
        auto this_config = config.get_table(config_key);
        
        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);

        // Time step
        oat::config::getValue(this_config, "dt", dt, 0.0);
        
        // Occlusion timeout
        double timeout_in_sec {0};
        if (oat::config::getValue(this_config, "timeout", timeout_in_sec, 0.0)) {
            not_found_count_threshold = static_cast<int>(timeout_in_sec / dt);
        }
        
        // Acceleration stdev
        oat::config::getValue(this_config, "sigma_accel", sig_accel, 0.0);
        
        // Measurement noise stdev
        oat::config::getValue(this_config, "sigma_noise", sig_measure_noise, 0.0);

        // GUI for tuning
        bool config_tune {false};
        oat::config::getValue(this_config, "tune", config_tune);
        
        if (config_tune) {
            tuning_on = true;
            createTuningWindows();
        }
        
    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

void KalmanFilter2D::initializeFilter(void) {

    initializeStaticMatracies();

    // Error covariance matrix (initialize with large value to indicate a lack
    // of trust in the model)
    cv::setIdentity(kf.errorCovPre, 1000.0);

    // TODO: Add head direction?
    // The state is
    // [ x  x'  y  y']^T, where ' denotes the time derivative
    // Initialize the state using the current measurement
    kf.statePre.at<double>(0) = kf_meas.at<double>(0);
    kf.statePre.at<double>(1) = 0.0;
    kf.statePre.at<double>(2) = kf_meas.at<double>(1);
    kf.statePre.at<double>(3) = 0.0;
}

void KalmanFilter2D::initializeStaticMatracies() {

    // State transition matrix
    // [ 1  dt 0  0  ]
    // [ 0  1  0  0  ]
    // [ 0  0  1  dt ]
    // [ 0  0  0  1  ]
    cv::setIdentity(kf.transitionMatrix);
    kf.transitionMatrix.at<double>(0, 1) = dt;
    kf.transitionMatrix.at<double>(2, 3) = dt;
    kf.transitionMatrix.at<double>(4, 5) = dt;

    // Observation Matrix (can only see position directly)
    // [ 1  0  0  0 ]
    // [ 0  0  1  0 ]
    cv::Mat::zeros(3, 6, CV_32F);
    kf.measurementMatrix.at<double>(0, 0) = 1.0;
    kf.measurementMatrix.at<double>(1, 2) = 1.0;

    // Noise covariance matrix (see pp13-15 of MWL.JPN.105.02.002 for derivation)
    // [ dt^4/4 dt^3/2 		     ] 
    // [ dt^3/2 dt^2   		     ]
    // [               dt^4/2 dt^2/3 ] * sigma_accel^2
    // [               dt^2/3 dt^2   ] 
    kf.processNoiseCov.at<double>(0, 0) = sig_accel * sig_accel * (dt * dt * dt * dt) / 4.0;
    kf.processNoiseCov.at<double>(0, 1) = sig_accel * sig_accel * (dt * dt * dt) / 2.0;
    kf.processNoiseCov.at<double>(0, 2) = 0.0;
    kf.processNoiseCov.at<double>(0, 3) = 0.0;
    
    kf.processNoiseCov.at<double>(1, 0) = sig_accel * sig_accel * (dt * dt * dt) / 2.0;
    kf.processNoiseCov.at<double>(1, 1) = sig_accel * sig_accel * (dt * dt);
    kf.processNoiseCov.at<double>(1, 2) = 0.0;
    kf.processNoiseCov.at<double>(1, 3) = 0.0;
    
    kf.processNoiseCov.at<double>(2, 0) = 0.0;
    kf.processNoiseCov.at<double>(2, 1) = 0.0;
    kf.processNoiseCov.at<double>(2, 2) = sig_accel * sig_accel * (dt * dt * dt * dt) / 4.0;
    kf.processNoiseCov.at<double>(2, 3) = sig_accel * sig_accel * (dt * dt * dt) / 2.0;
    
    kf.processNoiseCov.at<double>(3, 0) = 0.0;
    kf.processNoiseCov.at<double>(3, 1) = 0.0;
    kf.processNoiseCov.at<double>(3, 2) = sig_accel * sig_accel * (dt * dt * dt) / 2.0;
    kf.processNoiseCov.at<double>(3, 3) = sig_accel * sig_accel * (dt * dt);
    
    // Measurement noise covariance
    // [ sig_x^2  0 ]
    // [ 0  sig_y^2 ]
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(sig_measure_noise * sig_measure_noise));
}

void KalmanFilter2D::tune() {
    
    // TODO: The display output of this tuning feature is pretty useless. The constant
    // rescaling makes it very difficutl to get a sense of the absolute accuracy of 
    // filtering and how parameters affect this over time.
    if (tuning_on) {
        
        if (!tuning_windows_created) {
            createTuningWindows();
        }

        // Use the new parameters to create new static filter matracies
        sig_accel = ((double) sig_accel_tune);
        sig_measure_noise = ((double) sig_measure_noise_tune);
        initializeStaticMatracies();

        //cv::Mat tuning_canvas(canvas_hw, canvas_hw, CV_8UC3);
        //tuning_canvas.setTo(255);
        //drawPosition(tuning_canvas, raw_position);
        //drawPosition(tuning_canvas, filtered_position);

        // Draw the result, update sliders
        //cv::imshow(tuning_image_title, tuning_canvas);
        
        // If user hits escape, close the tuning windows
        char user_input;
        user_input = cv::waitKey(1);
        if (user_input == 27) { // Capture 'ESC' key
            tuning_on = false;
        }

    } else if (!tuning_on && tuning_windows_created) {
        // Destroy the tuning windows
        
        // TODO: Window will not actually close!!
        cv::destroyWindow(tuning_image_title);
        tuning_windows_created = false;

    }

}

void KalmanFilter2D::createTuningWindows() {

    // Create window for sliders
    cv::namedWindow(tuning_image_title, cv::WINDOW_AUTOSIZE);

    // Create sliders and insert them into window
    sig_accel_tune = (int)sig_accel;
    sig_measure_noise_tune = (int)sig_measure_noise;
    cv::createTrackbar("SIGMA ACCEL.", tuning_image_title, &sig_accel_tune, 1000);
    cv::createTrackbar("SIGMA NOISE", tuning_image_title, &sig_measure_noise_tune, 10);

    tuning_windows_created = true;
}

//void KalmanFilter2D::drawPosition(cv::Mat& canvas, const datatypes::Position2D& position) {
//
//    float x = position.position.x * draw_scale + (float) canvas_hw / 2.0;
//    float y = position.position.y * draw_scale + (float) canvas_hw / 2.0;
//    float dx = position.velocity.x * draw_scale;
//    float dy = position.velocity.y * draw_scale;
//
//    if (x > (canvas_hw - canvas_border)     ||
//            y > (canvas_hw - canvas_border) ||
//            x < canvas_border               ||
//            y < canvas_border)              {
//
//        draw_scale = draw_scale * 0.99;
//    } 
//
//    if (position.position_valid) {
//        cv::Point2f pos(x, y);
//        cv::circle(canvas, pos, 10, cv::Scalar(0, 0, 255), 2);
//    }
//
//    if (position.velocity_valid && position.position_valid) {
//        cv::Point2f pos(x, y);
//        cv::line(canvas, pos, cv::Point2f(x + dx, y + dy), cv::Scalar(0, 255, 0), 2, 8);
//    }
//}
