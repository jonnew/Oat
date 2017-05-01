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

#include <iostream>
#include <string>

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

po::options_description BackgroundSubtractor::options() const
{
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

    return local_opts;
}

void BackgroundSubtractor::applyConfiguration(const po::variables_map &vm,
                                              const config::OptionTable &config_table)
{
    // Background image path
    std::string img_path;
    if (oat::config::getValue(vm, config_table, "background", img_path)) {

        // TODO: Pixel color type check?
        background_mat_ = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

        if (background_mat_.data == nullptr)
            throw (std::runtime_error("File \"" + img_path + "\" could not be read."));

        background_set_ = true;
    }

    // Adaptation coefficient
    oat::config::getNumericValue<double>(vm, config_table, "adaptation-coeff", alpha_, 0.0, 1.0);
}

void BackgroundSubtractor::setBackgroundImage(const cv::Mat &mat)
{
    background_mat_ = mat.clone();
    mat.clone().convertTo(background_mat_f_, CV_32F);
    background_set_ = true;
}

void BackgroundSubtractor::filter(oat::Frame &frame)
{
    // This contains frame's data
    cv::Mat mat = frame.mat();

    // First image is always used as the default background image if one is
    // not provided in a configuration file
    if (!background_set_)
        setBackgroundImage(mat);

    if (alpha_ > 0.0) {
       cv::accumulateWeighted(mat, background_mat_f_, alpha_);
       background_mat_f_.convertTo(background_mat_, CV_8U);
    }

    mat -= background_mat_;
}

} /* namespace oat */
