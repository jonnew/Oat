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
, sig_accel(0.1) 
, sig_measure_noise (0.01) { 

	configureFilter();
}


KalmanFilter::configureFilter(void) {

	// The state is
	// TODO: Add head direction?
    // [x x' y y' z z'], where ' denotes the time derivative
    
    // State transition matrix
    // [ 1  dt 0  0  0  0 ]
    // [ 0  1  0  0  0  0 ]
    // [ 0  0  1  dt 0  0 ]
    // [ 0  0  0  1  0  0 ]
    // [ 0  0  0  0  1  dt]
    // [ 0  0  0  0  0  1 ]    
    cv::setIdentity(kf.transitionMatrix);
    kf.transitionMatrix.at<float>(1,2) = dt;
    kf.transitionMatrix.at<float>(3,4) = dt;
    kf.transitionMatrix.at<float>(5,6) = dt;
    
    
    // Observation Matrix (can only see position directly)
    // [ 1  0  0  0  0  0 ]
    // [ 0  0  1  0  0  0 ]
    // [ 0  0  0  0  1  0 ]
    cv::Mat::zeros(3, 6, CV_32F);
    kf.measurementMatrix.at<float>(1,1) = 1.0;
    kf.measurementMatrix.at<float>(2,3) = 1.0;
    kf.measurementMatrix.at<float>(3,5) = 1.0;
    
    // Noise covariance matrix (see pp13-15 of MWL.JPN.105.02.002 for derivation)
	// [dt^4/2 dt^2/3 							  ] 
	// [dt^2/3 dt^2   							  ]
	// [              dt^4/2 dt^2/3 			  ]
	// [              dt^2/3 dt^2  				  ] * sigma_accel^2
	// [                            dt^4/2 dt^2/3 ]
	// [                            dt^2/3 dt^2   ]
    kf.processNoiseCov.at<float>(1,1) = sig_accel*sig_accel * (dt*dt*dt*dt)/4.0;
    kf.processNoiseCov.at<float>(1,2) = sig_accel*sig_accel * (dt*dt*dt)/2.0
    kf.processNoiseCov.at<float>(1,3) = 0.0  
    kf.processNoiseCov.at<float>(1,4) = 0.0
    kf.processNoiseCov.at<float>(1,5) = 0.0
    kf.processNoiseCov.at<float>(1,6) = 0.0
    kf.processNoiseCov.at<float>(2,1) = sig_accel*sig_accel * (dt*dt*dt)/2.0
    kf.processNoiseCov.at<float>(2,2) = sig_accel*sig_accel * (dt*dt)
    kf.processNoiseCov.at<float>(2,3) = 0.0
    kf.processNoiseCov.at<float>(2,4) = 0.0
    kf.processNoiseCov.at<float>(2,5) = 0.0
    kf.processNoiseCov.at<float>(2,6) = 0.0
	kf.processNoiseCov.at<float>(3,1) = 0.0
	kf.processNoiseCov.at<float>(3,2) = 0.0
	kf.processNoiseCov.at<float>(3,3) = sig_accel*sig_accel * (dt*dt*dt*dt)/4.0; 
	kf.processNoiseCov.at<float>(3,4) = sig_accel*sig_accel * (dt*dt*dt)/2.0
	kf.processNoiseCov.at<float>(3,5) = 0.0
	kf.processNoiseCov.at<float>(3,6) = 0.0
	kf.processNoiseCov.at<float>(4,1) = 0.0
	kf.processNoiseCov.at<float>(4,2) = 0.0
	kf.processNoiseCov.at<float>(4,3) = sig_accel*sig_accel * (dt*dt*dt)/2.0
	kf.processNoiseCov.at<float>(4,4) = sig_accel*sig_accel * (dt*dt)
	kf.processNoiseCov.at<float>(4,5) = 0.0
	kf.processNoiseCov.at<float>(4,6) = 0.0
	kf.processNoiseCov.at<float>(5,1) = 0.0
	kf.processNoiseCov.at<float>(5,2) = 0.0
	kf.processNoiseCov.at<float>(5,3) = 0.0  
	kf.processNoiseCov.at<float>(5,4) = 0.0
	kf.processNoiseCov.at<float>(5,5) = sig_accel*sig_accel * (dt*dt*dt*dt)/4.0;
	kf.processNoiseCov.at<float>(5,6) = sig_accel*sig_accel * (dt*dt*dt)/2.0
	kf.processNoiseCov.at<float>(6,1) = 0.0
	kf.processNoiseCov.at<float>(6,2) = 0.0
	kf.processNoiseCov.at<float>(6,3) = 0.0
	kf.processNoiseCov.at<float>(6,4) = 0.0
	kf.processNoiseCov.at<float>(6,5) = sig_accel*sig_accel * (dt*dt*dt)/2.0
	kf.processNoiseCov.at<float>(6,6) = sig_accel*sig_accel * (dt*dt)
	
	// Meausrement noise covariance
	cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(sig_measure_noise * sig_measure_noise)));
}
