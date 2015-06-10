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

#include "FileReader.h"

#include <string>
#include <opencv2/videoio.hpp>

#include "../../lib/cpptoml/cpptoml.h"

FileReader::FileReader(std::string file_name_in, std::string image_sink_name, const double& frames_per_second) :
  Camera(image_sink_name)
, file_name(file_name_in)
, file_reader(file_name_in)
, use_roi(false)
, frame_rate_in_hz(frames_per_second) {

    // Default config
    configure();
}

void FileReader::grabMat() {
    
    file_reader >> current_frame;
    
    // Crop if necessary
    if (use_roi) {
        current_frame = current_frame(region_of_interest);
    }
}

bool FileReader::serveMat() {
    
    if (!current_frame.empty()) {
        frame_sink.pushMat(current_frame, current_sample);
        current_sample++;
        usleep(frame_period_in_us);
        return false;
    } else {
        frame_sink.set_running(false); //TODO: signal close somehow
        return true;
    }
}

void FileReader::configure() {
    calculateFramePeriod();
}

void FileReader::configure(std::string file_name, std::string key) {

    cpptoml::table config;

    try {
        config = cpptoml::parse_file(file_name);
    } catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse " << file_name << ": " << e.what() << std::endl;
    }

    try {
        // See if a camera configuration was provided
        if (config.contains(key)) {

            auto this_config = *config.get_table(key);

            if (this_config.contains("frame_rate")) {
                frame_rate_in_hz = (double) (*this_config.get_as<double>("frame_rate"));
                calculateFramePeriod();
            }
            
            if (this_config.contains("roi")) {

                auto roi = *this_config.get_table("roi");
                
                region_of_interest.x = (int) (*roi.get_as<int64_t>("x_offset"));
                region_of_interest.y = (int) (*roi.get_as<int64_t>("y_offset"));
                region_of_interest.width= (int) (*roi.get_as<int64_t>("width"));
                region_of_interest.height = (int) (*roi.get_as<int64_t>("height"));
                
                use_roi = true;

            } else {
                
                use_roi = false;
            }
            
            if (this_config.contains("calibration_file")) {
                
                std::string calibration_file = (*this_config.get_as<std::string>("calibration_file"));

                cv::FileStorage fs;
                fs.open(calibration_file, cv::FileStorage::READ);

                if (!fs.isOpened()) {
                    std::cerr << "Failed to open " << calibration_file << std::endl;
                    exit(EXIT_FAILURE);
                }

                // TODO: Exception handling for missing entries
                // Get calibration info
                fs["calibration_valid"] >> undistort_image;
                fs["camera_matrix"] >> camera_matrix;
                fs["distortion_coefficients"] >> distortion_coefficients;
                
                // Get homography info
                bool homography_valid;
                fs["homography_valid"] >> homography_valid;

                if (homography_valid) {

                    cv::Mat temp_mat;
                    fs["homography"] >> temp_mat;
                    cv::Matx33d homography((double*) temp_mat.clone().ptr());
                    frame_sink.set_homography(homography_valid, homography);
                }
                
                fs.release();
            }

        } else {
            std::cerr << "No HSV detector configuration named \"" + key + "\" was provided. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void FileReader::calculateFramePeriod() {
    frame_period_in_us = 1.0e6 * (1.0 / frame_rate_in_hz);
}
