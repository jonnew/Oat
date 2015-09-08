//******************************************************************************
//
//* File:   CameraCalibrator.cpp
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

#include "CameraCalibrator.h"

CameraCalibrator::CameraCalibrator(const std::string& frame_source_name, const CameraModel& model) :
  Calibrator(frame_source_name)
, calibration_valid_(false)
, model_(model)
{

    // if (interactive_) { // TODO: Generalize to accept images specifed by a file without interactive session

#ifdef OAT_USE_OPENGL
        try {
            cv::namedWindow(name(), cv::WINDOW_OPENGL & cv::WINDOW_KEEPRATIO);
        } catch (cv::Exception& ex) {
            oat::whoWarn(name(), "OpenCV not compiled with OpenGL support."
                    "Falling back to OpenCV's display driver.\n");
            cv::namedWindow(name(), cv::WINDOW_NORMAL & cv::WINDOW_KEEPRATIO);
        }
#else
        cv::namedWindow(name(), cv::WINDOW_NORMAL & cv::WINDOW_KEEPRATIO);
#endif

        std::cout << "Starting interactive session.\n";
        printUsage(std::cout);
   // }
}

void CameraCalibrator::configure(const std::string& config_file, const std::string& config_key) {

    // TODO: Provide list of image paths to perform calibraiton directly from file.

    // Available options
    std::vector<std::string> options {""};

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

    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

void CameraCalibrator::calibrate(cv::Mat& frame) {

    if (chessboard_detected_) {
        frame = drawCorners(frame, snapped_);
    }

    cv::imshow(name(), frame);
    char command = cv::waitKey(1);

    switch (command) {

        case 'f': // Change the calibration save path
        {
            changeSavePath();
            break;
        }
        case 'g': // Generate calibration parameters
        {
            generateCalibrationParameters();
            break;
        }
        case 'h': // Display help dialog
        {
            printUsage(std::cout);
            break;
        }
        case 'm': // Select homography estimation method
        {
            selectCalibrationMethod();
            break;
        }
        case 'p': // Print calibration results
        {
            printCalibrationResults(std::cout);
            break;
        }
        case 's': // Save homography info
        {
            saveCalibrationParameters();
            break;
        }
    }
}
