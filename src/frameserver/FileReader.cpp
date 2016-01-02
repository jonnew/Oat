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

#include <cpptoml.h>
#include "../../lib/utility/OatTOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

#include "FileReader.h"

namespace oat {

FileReader::FileReader(const std::string &image_sink_address,
                       const std::string &file_name,
                       const double frames_per_second) :
  FrameServer(image_sink_address)
, file_name_(file_name)
, file_reader_(file_name)
, frame_rate_in_hz_(frames_per_second)
{

    // Default config
    calculateFramePeriod();
    tick_ = clock_.now();
}

void FileReader::connectToNode() {

    // TODO: bind without using example frame from video stream. See PGGigECam.cpp
    // for example
    cv::Mat example_frame;
    file_reader_ >> example_frame;

    frame_sink_.bind(frame_sink_address_,
            example_frame.total() * example_frame.elemSize());

    shared_frame_ = frame_sink_.retrieve(
            example_frame.rows, example_frame.cols, example_frame.type());

    // Reset the video to the start
    file_reader_.set(CV_CAP_PROP_POS_AVI_RATIO, 0);

    // Put the sample rate in the shared frame
    shared_frame_.sample().set_period_sec(frame_period_in_sec_.count());
}

bool FileReader::serveFrame() {

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    frame_sink_.wait();

    // Acquire frame and increment sample count
    file_reader_ >> shared_frame_;
    shared_frame_.sample().incrementCount();

    // Crop if necessary
    if (use_roi_)
        shared_frame_ = shared_frame_(region_of_interest_);

    frame_empty_ = shared_frame_.empty();

    // Tell sources there is new data
    frame_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Enforce the correct frame rate
    std::this_thread::sleep_for(frame_period_in_sec_ - (clock_.now() - tick_));
    tick_ = clock_.now();

    return frame_empty_;
}

void FileReader::configure() { }

void FileReader::configure(const std::string& config_file, 
                           const std::string& config_key) {

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
        oat::config::getValue(this_config, "frame_rate", frame_rate_in_hz_, 0.0);
        calculateFramePeriod();

        // Set the ROI
        oat::config::Table roi;
        if (oat::config::getTable(this_config, "roi", roi)) {

            int64_t val;
            oat::config::getValue(roi, "x_offset", val, (int64_t)0, true);
            region_of_interest_.x = val;
            oat::config::getValue(roi, "y_offset", val, (int64_t)0, true);
            region_of_interest_.y = val;
            oat::config::getValue(roi, "width", val, (int64_t)0, true);
            region_of_interest_.width = val;
            oat::config::getValue(roi, "height", val, (int64_t)0, true);
            region_of_interest_.height = val;
            use_roi_ = true;
        }

    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

void FileReader::calculateFramePeriod() {

    std::chrono::duration<double> frame_period {1.0 / frame_rate_in_hz_};

    // Automatic conversion
    frame_period_in_sec_ = frame_period;
}

} /* namespace oat */
