//******************************************************************************
//* File:   Undistorter.h
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

#ifndef UNDISTORTER_H
#define	UNDISTORTER_H


#include "FrameFilter.h"

/**
 * Lens distortion compensation. 
 * 
 * Uses the results of oat-calibrate.
 */
class Undistorter : public FrameFilter {
public:

    /**
     * Lens distortion compensation. 
     * 
     * Uses the results of oat-calibrate. To reverse radial and tangential distortion
     * introduced by the camera lens and array mounting imperfections.
     * @param source_name raw frame source name
     * @param sink_name filtered frame sink name
     */
    Undistorter(const std::string& source_name, const std::string& sink_name);

    void configure(const std::string& config_file, const std::string& config_key) { } ;
    void loadCalibration(const std::string& calibration_file);
    
    // Accessors
    void set_camera_matrix(const cv::Matx33d& value) { camera_matrix_ = value; }
    void set_distortion_coefficients(const cv::Mat& value) { distortion_coefficients_ = value.clone(); }
    
private:
    
    /**
     * Apply undistortion.
     * @param frame unfiltered frame
     * @return filtered frame
     */
    cv::Mat filter(cv::Mat& frame);
    
    cv::Mat temp_matrix_;
    bool calibration_valid_ {false};
    cv::Matx33d camera_matrix_ ;
    cv::Mat distortion_coefficients_;

};

#endif	/* UNDISTORTER_H */

