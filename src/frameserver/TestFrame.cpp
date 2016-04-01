//******************************************************************************
//* File:   TestFrame.cpp
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
#include <thread>
#include <opencv2/core/core.hpp>
#include <cpptoml.h>

#include "../../lib/utility/OatTOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

#include "TestFrame.h"

namespace oat {

TestFrame::TestFrame(const std::string &image_sink_address,
                     const std::string &file_name,
                     const double frames_per_second) :
  FrameServer(image_sink_address)
, file_name_(file_name)
, frames_per_second_(frames_per_second)
{
    // Default config
    calculateFramePeriod();
    tick_ = clock_.now();
}

void TestFrame::configure(void) { }

void TestFrame::configure(const std::string &config_file,
                          const std::string &config_key) {

    // Available options
    std::vector<std::string> options {"num-samples",
                                      "fps"};

    // This will throw cpptoml::parse_exception if a file
    // with invalid TOML is provided
    auto config = cpptoml::parse_file(config_file);

    // See if a configuration was provided
    if (config->contains(config_key)) {

        // Get this components configuration table
        auto this_config = config->get_table(config_key);

        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);

        oat::config::getValue(this_config, "num-samples", num_samples_, 0);

        // Set the frame rate
        oat::config::getValue(this_config, "fps", frames_per_second_, 0.0);
        calculateFramePeriod();

    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

void TestFrame::connectToNode() {

    cv::Mat example_frame = cv::imread(file_name_);
    if (example_frame.empty())
        throw std::runtime_error(file_name_ + " could not be opened.");

    frame_sink_.bind(frame_sink_address_,
            example_frame.total() * example_frame.elemSize());

    shared_frame_ = frame_sink_.retrieve(
            example_frame.rows, example_frame.cols, example_frame.type());

    // Static image, never changes
    example_frame.copyTo(shared_frame_);

    // Put the sample rate in the shared frame
    shared_frame_.sample().set_rate_hz(1.0 / frame_period_in_sec_.count());
}

bool TestFrame::serveFrame() {

    if (it_ < num_samples_) {

        // START CRITICAL SECTION //
        ////////////////////////////

        // Wait for sources to read
        frame_sink_.wait();

        // Increment sample count
        shared_frame_.sample().incrementCount();

        // Tell sources there is new data
        frame_sink_.post();

        ////////////////////////////
        //  END CRITICAL SECTION  //

        it_++;

        std::this_thread::sleep_for(frame_period_in_sec_ - (clock_.now() - tick_));
        tick_ = clock_.now();

        return false;
    }

    return true;
}

void TestFrame::calculateFramePeriod() {

    std::chrono::duration<double> frame_period {1.0 / frames_per_second_};

    // Automatic conversion
    frame_period_in_sec_ = frame_period;
}

} /* namespace oat */
