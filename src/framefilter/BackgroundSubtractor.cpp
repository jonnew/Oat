//******************************************************************************
//* File:   BackgroundSubtractor.cpp
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

#include "BackgroundSubtractor.h"

#include <string>
#include <iostream>
#include <cpptoml.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/ProgramOptions.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

BackgroundSubtractor::BackgroundSubtractor(
            const std::string &frame_source_address,
            const std::string &frame_sink_address)
: FrameFilter(frame_source_address, frame_sink_address)
{
    // Nothing
}

void BackgroundSubtractor::appendOptions(po::options_description &opts)
{
    // Accepts a config file
    FrameFilter::appendOptions(opts);

    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("adaptation-coeff,a", po::value<double>(),
         "Scalar value, 0 to 1.0, specifying how quickly the new frames are "
         "used to update the backgound image. Default is 0, specifying no "
         "adaptation and a static background image that is never updated.")
        ("background,f", po::value<std::string>(),
         "Path to background image used for subtraction. If not provided, the "
         "first frame is used as the background image.")
        ;

    opts.add(local_opts);

    // Return valid keys
    for (auto &o: local_opts.options())
        config_keys_.push_back(o->long_name());
}

void BackgroundSubtractor::configure(const po::variables_map &vm)
{
    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Background image path
    std::string img_path;
    if (oat::config::getValue(vm, config_table, "background", img_path)) {

        // TODO: Color image only?
        background_frame_ = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

        if (background_frame_.data == NULL)
            throw (std::runtime_error("File \"" + img_path + "\" could not be read."));

        background_set_ = true;
    }

    // Adaptation coefficient
    oat::config::getNumericValue<double>(vm, config_table, "adaptation-coeff", alpha_, 0.0, 1.0);
}

void BackgroundSubtractor::setBackgroundImage(const cv::Mat &frame)
{
    background_frame_ = frame.clone();
    frame.clone().convertTo(background_frame_f_, CV_32F);
    background_set_ = true;
}

void BackgroundSubtractor::filter(cv::Mat &frame)
{
    // First image is always used as the default background image if one is
    // not provided in a configuration file
    if (!background_set_)
        setBackgroundImage(frame);

    if (alpha_ > 0.0) {
       cv::accumulateWeighted(frame, background_frame_f_, alpha_);
       background_frame_f_.convertTo(background_frame_, CV_8U);
    }

    frame = frame - background_frame_;
}

} /* namespace oat */
