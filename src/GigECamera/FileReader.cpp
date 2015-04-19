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
#include <opencv2/opencv.hpp>

#include "../../lib/cpptoml/cpptoml.h"

FileReader::FileReader(std::string file_name_in, std::string image_sink_name) :
Camera(image_sink_name)
, file_name(file_name_in)
, file_reader(file_name_in) {

    // Default config
    configure();
}

void FileReader::grabMat() {
    file_reader >> current_frame;
}

void FileReader::serveMat() {
    if (!current_frame.empty()) {
        frame_sink.set_shared_mat(current_frame);
        usleep(frame_period_in_us);
    }
}

void FileReader::configure() {
    frame_rate_in_hz = 24;
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
