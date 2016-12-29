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

#include <string>
#include <opencv2/opencv.hpp>
#include <cpptoml.h>

#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

#include "KalmanFilter2D.h"

namespace oat {

KalmanFilter2D::KalmanFilter2D(const std::string& position_source_address,
                               const std::string& position_sink_address) :
  PositionFilter(position_source_address, position_sink_address)
, tuning_image_title_(position_sink_address + "_tuning")
{
    sig_accel_tune_ = static_cast<int>(sig_measure_noise_);
    sig_measure_noise_tune_ = static_cast<int>(sig_measure_noise_);
}

po::options_description KalmanFilter2D::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("dt", po::value<double>(), // TODO: should be automatic?
         "Kalman filter time step in seconds.")
        ("timeout,T", po::value<double>(),
         "Seconds to perform position estimation detection with lack of "
         "position measure. Defaults to 0.")
        ("sigma-accel,a", po::value<double>(), // TODO: Should be specified in each dimension
         "Standard deviation of normally distributed, random accelerations used "
         "by the internal model of object motion (position units/s2; e.g. "
         "pixels/s2).")
        ("sigma-noise,n", po::value<double>(), // TODO: Should be specified in each dimension
         "Standard deviation of randomly distributed position measurement noise "
         "(position units; e.g. pixels).")
        ("tune,t",
         "If true, provide a GUI with sliders for tuning filter "
         "parameters.")
        ;

    return local_opts;
}

void KalmanFilter2D::applyConfiguration(const po::variables_map &vm,
                                        const config::OptionTable &config_table)
{
    // Time step
    oat::config::getNumericValue<double>(vm, config_table, "dt", dt_, 0);

    // Blind filter timeout
    double t;
    if (oat::config::getNumericValue<double>(vm, config_table, "timeout", t, 0))
        not_found_count_threshold_ = static_cast<int>(t / dt_);

    // Sigma accel
    oat::config::getNumericValue<double>(
        vm, config_table, "sigma-accel", sig_accel_, 0);

    // Sigma noise
    oat::config::getNumericValue<double>(
        vm, config_table, "sigma-noise", sig_measure_noise_, 0);

    // Tuning GUI
    bool tune;
    oat::config::getValue<bool>(vm, config_table, "tune", tune);

    if (tune) {
        tuning_on_ = true;
        createTuningWindows();
    }
}

void KalmanFilter2D::filter(oat::Position2D &position) {

    // Transform raw position into kf_meas_ vector
    if (position.position_valid) {
        kf_meas_(0,0) = position.position.x;
        kf_meas_(1,0) = position.position.y;
        not_found_count_ = 0;

        // We are coming from a time step where there were no measurements for
        // a long time, or the first sample, so we need to reinitialize the
        // filter
        if (!found_)
            initializeFilter();

        found_ = true;
    } else {
        not_found_count_++;
    }

    // If we have not gotten a measurement of the object for a long time
    // we need to reinitialize the filter
    if (not_found_count_ >= not_found_count_threshold_)
        found_ = false;

    // Only update if the object is found_ (this includes time points for which
    // the position measurement was invalid, but we are within the not_found_count_threshold_)
    if (found_) {

        kf_predicted_state_ = kf_.predict();

        // Apply the Kalman update
        kf_.correct(kf_meas_);
    }

    position.position.x = kf_predicted_state_(0, 0);
    position.velocity.x = kf_predicted_state_(1, 0);
    position.position.y = kf_predicted_state_(2, 0);
    position.velocity.y = kf_predicted_state_(3, 0);

    // This Position is only valid if the not_found_count_threshold_ has not
    // be exceeded
    if (found_) {
        position.position_valid = true;
        position.velocity_valid = true;
    } else {
        position.position_valid = false;
        position.velocity_valid = false;
    }

    // Tune the filter, if requested
    tune();
}

void KalmanFilter2D::initializeFilter(void) {

    initializeStaticMatracies();

    // Error covariance matrix (initialize with large value to indicate a lack
    // of trust in the model)
    cv::setIdentity(kf_.errorCovPre, 1000.0);

    // TODO: Add head direction?
    // The state is
    // [ x  x'  y  y']^T, where ' denotes the time derivative
    // Initialize the pre/post state using the current measurement
    kf_.statePre.at<double>(0) = kf_meas_(0, 0);
    kf_.statePre.at<double>(1) = 0.0;
    kf_.statePre.at<double>(2) = kf_meas_(1, 0);
    kf_.statePre.at<double>(3) = 0.0;

    kf_.statePost.at<double>(0) = kf_meas_(0, 0);
    kf_.statePost.at<double>(1) = 0.0;
    kf_.statePost.at<double>(2) = kf_meas_(1, 0);
    kf_.statePost.at<double>(3) = 0.0;
}

void KalmanFilter2D::initializeStaticMatracies() {

    // State transition matrix
    // [ 1  dt_ 0  0  ]
    // [ 0  1  0  0   ]
    // [ 0  0  1  dt_ ]
    // [ 0  0  0  1   ]
    cv::setIdentity(kf_.transitionMatrix);
    kf_.transitionMatrix.at<double>(0, 1) = dt_;
    kf_.transitionMatrix.at<double>(2, 3) = dt_;

    // Observation Matrix (can only see position directly)
    // [ 1  0  0  0 ]
    // [ 0  0  1  0 ]
    kf_.measurementMatrix = cv::Mat::zeros(2, 4, CV_64F);
    kf_.measurementMatrix.at<double>(0, 0) = 1.0;
    kf_.measurementMatrix.at<double>(1, 2) = 1.0;

    // Noise covariance matrix (see pp13-15 of MWL.JPN.105.02.002 for derivation)
    // [ dt_^4/4 dt_^3/2 		       ]
    // [ dt_^3/2 dt_^2   		       ]
    // [               dt_^4/2 dt_^2/3 ] * sigma_accel^2
    // [               dt_^2/3 dt_^2   ]
    kf_.processNoiseCov = cv::Mat::zeros(4, 4, CV_64F);
    kf_.processNoiseCov.at<double>(0, 0) = sig_accel_ * sig_accel_ * (dt_ * dt_ * dt_ * dt_) / 4.0;
    kf_.processNoiseCov.at<double>(0, 1) = sig_accel_ * sig_accel_ * (dt_ * dt_ * dt_) / 2.0;

    kf_.processNoiseCov.at<double>(1, 0) = sig_accel_ * sig_accel_ * (dt_ * dt_ * dt_) / 2.0;
    kf_.processNoiseCov.at<double>(1, 1) = sig_accel_ * sig_accel_ * (dt_ * dt_);

    kf_.processNoiseCov.at<double>(2, 2) = sig_accel_ * sig_accel_ * (dt_ * dt_ * dt_ * dt_) / 4.0;
    kf_.processNoiseCov.at<double>(2, 3) = sig_accel_ * sig_accel_ * (dt_ * dt_ * dt_) / 2.0;

    kf_.processNoiseCov.at<double>(3, 2) = sig_accel_ * sig_accel_ * (dt_ * dt_ * dt_) / 2.0;
    kf_.processNoiseCov.at<double>(3, 3) = sig_accel_ * sig_accel_ * (dt_ * dt_);

    // Measurement noise covariance
    // [ sig_x^2  0 ]
    // [ 0  sig_y^2 ]
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar(sig_measure_noise_ * sig_measure_noise_));
}

void KalmanFilter2D::tune() {

    // TODO: The display output of this tuning feature is pretty useless. The constant
    // rescaling makes it very difficult to get a sense of the absolute accuracy of
    // filtering and how parameters affect this over time.
    if (tuning_on_) {

        if (!tuning_windows_created_) {
            createTuningWindows();
        }

        // Use the new parameters to create new static filter matracies
        sig_accel_ = static_cast<double>(sig_accel_tune_);
        sig_measure_noise_ = static_cast<double>(sig_measure_noise_tune_);
        initializeStaticMatracies();

        //cv::Mat tuning_canvas(canvas_hw, canvas_hw, CV_8UC3);
        //tuning_canvas.setTo(255);
        //drawPosition(tuning_canvas, raw_position);
        //drawPosition(tuning_canvas, filtered_position);

        // Draw the result, update sliders
        //cv::imshow(tuning_image_title_, tuning_canvas);

        // If user hits escape, close the tuning windows
        char user_input;
        user_input = cv::waitKey(1);
        if (user_input == 27) { // Capture 'ESC' key
            tuning_on_ = false;
        }

    } else if (!tuning_on_ && tuning_windows_created_) {
        // Destroy the tuning windows

        // TODO: Window will not actually close!!
        cv::destroyWindow(tuning_image_title_);
        tuning_windows_created_ = false;

    }
}

void KalmanFilter2D::createTuningWindows() {

    // Create window for sliders
    cv::namedWindow(tuning_image_title_, cv::WINDOW_AUTOSIZE);

    // Create sliders and insert them into window
    sig_accel_tune_ = static_cast<int>(sig_accel_);
    sig_measure_noise_tune_ = static_cast<int>(sig_measure_noise_);
    cv::createTrackbar("SIGMA ACCEL.", tuning_image_title_, &sig_accel_tune_, 1000);
    cv::createTrackbar("SIGMA NOISE", tuning_image_title_, &sig_measure_noise_tune_, 10);

    tuning_windows_created_ = true;
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

} /* namespace oat */
