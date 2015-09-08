//******************************************************************************
//* File:   CameraCalibrator.h
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
//****************************************************************************

#ifndef CAMERACALIBRATOR_H
#define CAMERACALIBRATOR_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "Calibrator.h"

/**
 * Interactive camera calibration and distortion parameter generator.
 */
class CameraCalibrator : public Calibrator {

public:
    
    // Camera model to use for calibration
    enum class CameraModel { PINHOLE = 0, FISHEYE };

    /**
     * Interactive camera calibrator. The corners of a chessboard pattern (of a
     * user-specified size and element length) are automatically detected in
     * the frame stream. These points are gathered, and upon the user's
     * request, used to generate a camera matrix and lens distortion
     * coefficient set. Two camera models are available: pinhole and fisheye.
     * @param frame_source_name imaging setup frame source name 
     * @param model Camera model used to generate camera matrix and distortion coefficients.
     */
    CameraCalibrator(const std::string& frame_source_name, const CameraModel& model);

    /**
     * Configure calibration parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    void configure(const std::string& config_file, const std::string& config_key) override;

protected:

    /**
     * Perform camera calibration routine.
     * @param frame current frame to use for running calibration
     */
    void calibrate(cv::Mat& frame) override;

private:

    // Is homography well-defined?
    bool calibration_valid_;
    cv::Mat camera_matrix_, distortion_coefficients_;

    // Default esimation method
    CameraModel model_ {CameraModel::PINHOLE};

    // NXM black squares in the chessboard
    bool chessboard_detected_, snapped_ {false};
    cv::Size chessboard_size_;

    // Minimum time between chessboard corner detections
    double detection_delay_sec_ {1.0};

    // Data used to create homography    
    std::vector<std::vector<cv::Point2f>> corners_; 
    
    // Interactive session 
    int detectCorners(void);
    void printDataPoints(void);
    void printCalibrationResults(void);
    int generateCalibrationParameters(void);
    int selectCalibrationMethod(void);
    int saveCalibrationParameters(void);
    cv::Mat drawCorners(cv::Mat& frame, bool invert_colors);
    
};

#endif //CAMERACALIBRATOR_H

