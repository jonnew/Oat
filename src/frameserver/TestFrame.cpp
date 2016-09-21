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

#include "TestFrame.h"

#include <thread>

#include <cpptoml.h>

#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

namespace oat {

TestFrame::TestFrame(const std::string &sink_address)
: FrameServer(sink_address)
{
    // Initialize time
    tick_ = clock_.now();
}

void TestFrame::appendOptions(po::options_description &opts)
{
    // Accepts default options
    FrameServer::appendOptions(opts);

    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("test-image,f", po::value<std::string>(),
         "Path to test image used as frame source.")
        ("color,C", po::value<std::string>(),
         "Pixel color format. Defaults to BGR.\n"
         "Values:\n"
         "  GREY: \t 8-bit Greyscale image.\n"
         "  BGR: \t8-bit, 3-chanel, BGR Color image.\n")
        ("fps,r", po::value<double>(),
         "Frames to serve per second.")
        ("num-frames,n", po::value<uint64_t>(),
         "Number of frames to serve before exiting.")
        ;

    opts.add(local_opts);

    // Return valid keys
    for (auto &o: local_opts.options())
        config_keys_.push_back(o->long_name());
}

void TestFrame::configure(const po::variables_map &vm) {

    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Test image path
    oat::config::getValue(vm, config_table, "test-image", file_name_, true);

    // Pixel color
    std::string col;
    if (oat::config::getValue<std::string>(vm, config_table, "color", col)) {
        color_ = oat::str_color(col);
    }

    // Number of frames to serve
    oat::config::getNumericValue<uint64_t>(
        vm, config_table, "num-frames", num_samples_, 1
    );

    // Frame rate
    if (oat::config::getNumericValue(vm, config_table, "fps", frames_per_second_, 0.0))
        calculateFramePeriod();
}

void TestFrame::connectToNode() {

    auto mat = cv::imread(file_name_, oat::imread_code(color_));

    if (mat.data == NULL)
        throw (std::runtime_error("File \"" + file_name_ + "\" could not be read."));

    frame_sink_.bind(frame_sink_address_,
            mat.total() * mat.elemSize());

    shared_frame_ = frame_sink_.retrieve(
            mat.rows, mat.cols, mat.type(), color_);

    // Static image, never changes
    mat.copyTo(shared_frame_);

    // Put the sample rate in the shared frame
    shared_frame_.set_rate_hz(1.0 / frame_period_in_sec_.count());
}

bool TestFrame::process()
{
    if (shared_frame_.sample_count() < num_samples_) {

        // START CRITICAL SECTION //
        ////////////////////////////

        // Wait for sources to read
        frame_sink_.wait();

        // Zero frame copy
        shared_frame_.incrementSampleCount();

        // Tell sources there is new data
        frame_sink_.post();

        ////////////////////////////
        //  END CRITICAL SECTION  //

        std::this_thread::sleep_for(frame_period_in_sec_ - (clock_.now() - tick_));
        tick_ = clock_.now();

        return false;
    }

    return true;
}

void TestFrame::calculateFramePeriod()
{
    std::chrono::duration<double> frame_period {1.0 / frames_per_second_};

    // Automatic conversion
    frame_period_in_sec_ = frame_period;
}

} /* namespace oat */
