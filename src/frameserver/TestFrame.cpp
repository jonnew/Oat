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
#include <opencv2/core/core.hpp>
#include <cpptoml.h>

#include "../../lib/utility/OatTOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

#include "TestFrame.h"

namespace oat {

TestFrame::TestFrame(const std::string &image_sink_address,
                     const std::string &file_name) :
  FrameServer(image_sink_address)
, file_name_(file_name)
{
    // Nothing
}

void TestFrame::configure(void) { }

void TestFrame::configure(const std::string &config_file,
                          const std::string &config_key) {

    // Available options
    std::vector<std::string> options {"num-samples"};

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

    // Put a dummy rate in the shared frame
    shared_frame_.sample().set_rate_hz(1000000.0);
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

        return false;
    }

    return true;
}

} /* namespace oat */
