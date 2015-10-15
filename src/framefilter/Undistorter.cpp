//******************************************************************************
//* File:   Undistorter.cpp
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

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "Undistorter.h"

Undistorter::Undistorter(const std::string& source_name, const std::string& sink_name) :
  FrameFilter(source_name, sink_name)
, camera_matrix_(cv::Matx33d::eye())
, distortion_coefficients_ (cv::Mat::zeros(8, 1, CV_64F)) 
{
    // Nothing
}

void Undistorter::loadCalibration(const std::string& calibration_file) {
    
    
    
}


cv::Mat Undistorter::filter(cv::Mat& frame) {
    
    frame.copyTo(temp_matrix_);
    cv::undistort(temp_matrix_, frame, camera_matrix_, distortion_coefficients_);
    return frame;
}

