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

#include "TestPosition.h"

#include <string>
#include <iostream>
#include <limits>
#include <math.h>
#include <opencv2/opencv.hpp>

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/shmem/MatServer.h"

TestPosition::TestPosition(std::string position_sink_name) :
  position_sink(position_sink_name)
, accel_distribution(0.0, 5.0)
, state(6,1,CV_32F)
, accel_vec(3,1,CV_32F)
, state_transition_mat(6, 6, CV_32F)
, input_mat(6, 3, CV_32F) {

    position_sink.createSharedObject();
    createStaticMatracies();
    
    // Initial condition
    state.at<float>(0) = 0.0; // x
    state.at<float>(1) = 0.0; // x'
    state.at<float>(2) = 0.0; // y
    state.at<float>(3) = 0.0; // y'
    state.at<float>(4) = 0.0; // z
    state.at<float>(5) = 0.0; // z'

}

void TestPosition::simulateAndServePosition() {
    
    // Simulate one step of random, but smooth, motion
    simulateMotion(); 
    
    // Transform into shmem::Positions type
    shmem::Position pos;
    
    // Simulated position info
    pos.position_valid = true;
    pos.position.x = state.at<float>(0);
    pos.position.y = state.at<float>(2);
    pos.position.z = state.at<float>(4);
    
    // We have access to the velocity info for comparison
    pos.velocity_valid = true;
    pos.velocity.x = state.at<float>(1);
    pos.velocity.y = state.at<float>(3);
    pos.velocity.z = state.at<float>(5);
    
    // Publish simulated position
    position_sink.pushObject(pos);
    
}


void TestPosition::simulateMotion() {
    
    // Generate random acceleration
    accel_vec.at<float>(0) = accel_distribution(accel_generator);
    accel_vec.at<float>(1) = accel_distribution(accel_generator);
    accel_vec.at<float>(2) = accel_distribution(accel_generator);
    
    // Apply acceleration and transition matrix to the simulated position
    state = 
        state_transition_mat * state + input_mat * accel_vec;
}

void TestPosition::createStaticMatracies() {
    
    // State transition matrix
    state_transition_mat.at<float>(0, 0) = 1.0;
    state_transition_mat.at<float>(0, 1) = DT;
    state_transition_mat.at<float>(0, 2) = 0.0;
    state_transition_mat.at<float>(0, 3) = 0.0;
    state_transition_mat.at<float>(0, 3) = 0.0;
    state_transition_mat.at<float>(0, 3) = 0.0;
    
    state_transition_mat.at<float>(1, 0) = 0.0;
    state_transition_mat.at<float>(1, 1) = 1.0;
    state_transition_mat.at<float>(1, 2) = 0.0;
    state_transition_mat.at<float>(1, 3) = 0.0;
    state_transition_mat.at<float>(1, 3) = 0.0;
    state_transition_mat.at<float>(1, 3) = 0.0;
    
    state_transition_mat.at<float>(2, 0) = 0.0;
    state_transition_mat.at<float>(2, 1) = 0.0;
    state_transition_mat.at<float>(2, 2) = 1.0;
    state_transition_mat.at<float>(2, 3) = DT;
    state_transition_mat.at<float>(2, 3) = 0.0;
    state_transition_mat.at<float>(2, 3) = 0.0;
    
    state_transition_mat.at<float>(3, 0) = 0.0;
    state_transition_mat.at<float>(3, 1) = 0.0;
    state_transition_mat.at<float>(3, 2) = 0.0;
    state_transition_mat.at<float>(3, 3) = 1.0;
    state_transition_mat.at<float>(3, 3) = 0.0;
    state_transition_mat.at<float>(3, 3) = 0.0;
    
    state_transition_mat.at<float>(4, 0) = 0.0;
    state_transition_mat.at<float>(4, 1) = 0.0;
    state_transition_mat.at<float>(4, 2) = 0.0;
    state_transition_mat.at<float>(4, 3) = 0.0;
    state_transition_mat.at<float>(4, 3) = 1.0;
    state_transition_mat.at<float>(4, 3) = DT;
    
    state_transition_mat.at<float>(5, 0) = 0.0;
    state_transition_mat.at<float>(5, 1) = 0.0;
    state_transition_mat.at<float>(5, 2) = 0.0;
    state_transition_mat.at<float>(5, 3) = 0.0;
    state_transition_mat.at<float>(5, 3) = 0.0;
    state_transition_mat.at<float>(5, 3) = 1.0;
    
    // Input Matrix
    input_mat.at<float>(0, 0) = (DT*DT)/2;
    input_mat.at<float>(0, 1) = 0.0;
    input_mat.at<float>(0, 2) = 0.0;
    
    input_mat.at<float>(1, 0) = DT;
    input_mat.at<float>(1, 1) = 0.0;
    input_mat.at<float>(1, 2) = 0.0;
    
    input_mat.at<float>(2, 0) = 0.0;
    input_mat.at<float>(2, 1) = (DT*DT)/2;
    input_mat.at<float>(2, 2) = 0.0;
    
    input_mat.at<float>(3, 0) = 0.0;
    input_mat.at<float>(3, 1) = DT;
    input_mat.at<float>(3, 2) = 0.0;
    
    input_mat.at<float>(4, 0) = 0.0;
    input_mat.at<float>(4, 1) = 0.0;
    input_mat.at<float>(4, 2) = (DT*DT)/2;
    
    input_mat.at<float>(5, 0) = 0.0;
    input_mat.at<float>(5, 1) = 0.0;
    input_mat.at<float>(5, 2) = DT;

}
