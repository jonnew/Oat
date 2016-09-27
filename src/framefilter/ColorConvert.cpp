//******************************************************************************
//* File:   ColorConvert.cpp
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

#include "ColorConvert.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/ProgramOptions.h"
#include "../../lib/utility/TOMLSanitize.h"

// Populate available conversion

namespace oat {

ColorConvert::ColorConvert(const std::string &frame_source_address,
                           const std::string &frame_sink_address)
: FrameFilter(frame_source_address, frame_sink_address)
{
    // Nothing
}

void ColorConvert::appendOptions(po::options_description &opts)
{
    // Accepts a config file
    FrameFilter::appendOptions(opts);

    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("color,C", po::value<std::string>(),
         "Pixel color format." 
         "Values:\n"
         "  GREY: \t 8-bit Greyscale image.\n"
         "  BRG: \t8-bit, 3-chanel, BGR Color image.\n"
         "  HSV: \t8-bit, 3-chanel, HSV Color image.\n")
        ;

    opts.add(local_opts);

    // Return valid keys
    for (auto &o : local_opts.options())
        config_keys_.push_back(o->long_name());
}

void ColorConvert::configure(const po::variables_map &vm)
{
    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Pixel color to convert to
    std::string col;
    if (oat::config::getValue<std::string>(
            vm, config_table, "color", col, true)) {
        color_ = oat::str_color(col);
    }
}

void ColorConvert::connectToNode()
{
    // Establish our a slot in the source node
    frame_source_.touch(frame_source_address_);

    // Wait for synchronous start with sink when it binds its node
    frame_source_.connect();

    // Get frame meta data to format sink
    auto frame_parameters = frame_source_.parameters();

    // Get the color conversion code
    conversion_code_ = oat::color_conv_code(frame_parameters.color, color_);

    // If there is no conversion being done, throw
    if (conversion_code_ == -1) {
        throw std::runtime_error("Nothing to be done for " + color_str(frame_parameters.color)
                                 + " to "
                                 + color_str(color_)
                                 + " conversion.");
    }

    // Bind to sink node and create a shared frame
    // Because this changes the color, it might change the size and type of
    // frame
    size_t bytes = frame_parameters.rows * frame_parameters.cols
                   * oat::color_bytes(color_);
    frame_sink_.bind(frame_sink_address_, bytes);
    shared_frame_ = frame_sink_.retrieve(frame_parameters.rows,
                                         frame_parameters.cols,
                                         oat::cv_type(color_),
                                         color_);
}

void ColorConvert::filter(cv::Mat &frame)
{
    cv::Mat out; // Might change underlying element type
    cv::cvtColor(frame, out, conversion_code_);
    frame = out;
    static_cast<oat::Frame &>(frame).set_color(color_);
}

} /* namespace oat */

