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

#ifndef TESTPOSITION_H
#define	TESTPOSITION_H

#include <string>
#include <random>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmem/Position.h"
#include "../../lib/shmem/SMServer.h"

#define DT 0.02 // TODO: Config

class TestPosition  {
    
public:
    
    TestPosition(std::string pos_sink_name);

    // Use a configuration file to specify parameters
    void configure(std::string file_name, std::string key);
    
    // Simulate object position motion and publish to shared memory
    void simulateAndServePosition(void);
    
    void stop(void) { position_sink.set_running(false); }
    
private:
    
    // Random number generator
    std::default_random_engine accel_generator;
    std::normal_distribution<float> accel_distribution;
    
    // Simulated 3D position
    cv::Mat state;
    cv::Mat accel_vec;
    
    // STM and input matrix
    cv::Mat state_transition_mat;
    cv::Mat input_mat;
    
    shmem::SMServer<shmem::Position> position_sink;
    
    void createStaticMatracies(void);
    void simulateMotion(void);
    
};

#endif	/* TESTPOSITION_H */

