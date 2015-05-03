//******************************************************************************
//* File:   %<%NAME%>%.%<%EXTENSION%>%
//* Author: %<%USER%>%
//
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
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

#ifndef RANDOMACCEL2D
#define	RANDOMACCEL2D


#include <string>

#include "TestPosition.h"

class RandomAccel2D : public TestPosition {
    
public:

    RandomAccel2D(std::string position_sink_name);

    //void configure(std::string file_name, std::string key);
    void simulateAndServePosition(void);

private:

    // Random number generator
    std::default_random_engine accel_generator;
    std::normal_distribution<double> accel_distribution;

    // Simulated position
    cv::Matx41d state;
    cv::Matx21d accel_vec;

    // STM and input matrix
    cv::Matx44d state_transition_mat;
    cv::Matx<double, 4, 2> input_mat;

    void createStaticMatracies(void);
    void simulateMotion(void);

};

#endif	/* RANDOMACCEL2D */

