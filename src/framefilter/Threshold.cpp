//******************************************************************************
//* File:   Threshold.cpp
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

#include "Threshold.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/ProgramOptions.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

Threshold::Threshold(const std::string &frame_source_address,
                     const std::string &frame_sink_address)
: FrameFilter(frame_source_address, frame_sink_address)
{
    // Nothing
}

po::options_description Threshold::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("intensity,I", po::value<std::string>(),
         "Array of ints between 0 and 256, [min,max], specifying the "
         "intensity passband.")
        ;

    return local_opts;
}

void Threshold::applyConfiguration(const po::variables_map &vm,
                                   const config::OptionTable &config_table)
{
    // Intensity
    std::vector<int> i;
    if (oat::config::getArray<int, 2>(vm, config_table, "intensity", i)) {

        i_min_ = i[0];
        i_max_ = i[1];

        if (i_min_ < 0 || i_min_> 256 || i_max_ < 0 || i_max_ > 256)
           throw std::runtime_error("Values of intensity should be between 0 and 256.");
    }
}

void Threshold::filter(cv::Mat &frame)
{
    cv::Mat grey_frame, thresh_frame;

    auto conversion_code = oat::color_conv_code(
        static_cast<oat::Frame &>(frame).color(), oat::PIX_GREY);

    if (conversion_code >= 0)
        cv::cvtColor(frame, grey_frame, conversion_code);
    else
        grey_frame = frame;

    cv::inRange(grey_frame, i_min_, i_max_, thresh_frame);
    frame.setTo(cv::Scalar(0, 0, 0), thresh_frame == 0);
}

} /* namespace oat */
