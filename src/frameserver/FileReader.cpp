//******************************************************************************
//* File:   FileReader.cpp
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

#include <chrono>
#include <string>
#include <thread>
#include <opencv2/videoio.hpp>

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/cpptoml/OatTOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

#include "FileReader.h"

FileReader::FileReader(std::string file_name_in, 
        std::string image_sink_name, 
        const double frames_per_second) :
  FrameServer(image_sink_name)
, file_name(file_name_in)
, file_reader(file_name_in)
, use_roi(false)
, frame_rate_in_hz(frames_per_second) {

    // Default config
    configure();
    tick = clock.now();
}

void FileReader::grabFrame(cv::Mat& frame) {
    
    file_reader >> frame;
    
    // Crop if necessary
    if (use_roi) {
        frame = frame(region_of_interest);
    }
    
    auto tock = clock.now();
    std::this_thread::sleep_for(frame_period_in_sec - (tock - tick));
    
    tick = clock.now();
}

void FileReader::configure() {
    calculateFramePeriod();
}

void FileReader::configure(const std::string& config_file, const std::string& config_key) {

    // Available options
    std::vector<std::string> options {"frame_rate", "roi"};

    // This will throw cpptoml::parse_exception if a file 
    // with invalid TOML is provided
    auto config = cpptoml::parse_file(config_file);

    // See if a camera configuration was provided
    if (config->contains(config_key)) {
        
        // Get this components configuration table
        auto this_config = config->get_table(config_key);
        
        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);
        
        // Set the frame rate
        oat::config::getValue(this_config, "frame_rate", frame_rate_in_hz, 0.0);
        calculateFramePeriod();

        // Set the ROI
        oat::config::Table roi;
        if (oat::config::getTable(this_config, "roi", roi)) {

            int64_t val;
            oat::config::getValue(roi, "x_offset", val, (int64_t)0, true);
            region_of_interest.x = val;
            oat::config::getValue(roi, "y_offset", val, (int64_t)0, true);
            region_of_interest.y = val;
            oat::config::getValue(roi, "width", val, (int64_t)0, true);
            region_of_interest.width = val;
            oat::config::getValue(roi, "height", val, (int64_t)0, true);
            region_of_interest.height = val;
            use_roi = true;

        } else {
            use_roi = false;
        }

        // TODO: Exception handling for missing entries
        // Get calibration info
        // TODO: use standard TOML format for these matracies instead 
        // of the secondary YML config file
        std::string calibration_file;
        if (oat::config::getValue(this_config, "calibration_file", calibration_file)) {

            cv::FileStorage fs;
            fs.open(calibration_file, cv::FileStorage::READ);

            if (!fs.isOpened()) {
                throw (std::runtime_error("Failed to open calibration file " + calibration_file));
            }

            fs["calibration_valid"] >> undistort_image;
            fs["camera_matrix"] >> camera_matrix;
            fs["distortion_coefficients"] >> distortion_coefficients;

            fs.release();
        }
    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

void FileReader::calculateFramePeriod() {
    
    std::chrono::duration<double> frame_period {1.0 / frame_rate_in_hz};

    // Automatic conversion
    frame_period_in_sec = frame_period;
}
