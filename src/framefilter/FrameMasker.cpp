//******************************************************************************
//* File:   FrameMasker.cpp
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

#include "FrameMasker.h"

#include <cpptoml.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/ProgramOptions.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

FrameMasker::FrameMasker(const std::string &frame_source_address,
                         const std::string &frame_sink_address)
: FrameFilter(frame_source_address, frame_sink_address)
{
    // Nothing
}

po::options_description FrameMasker::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("mask,f", po::value<std::string>(),
         "Path to a binary image used to mask frames from SOURCE. SOURCE frame "
         "pixels with indices corresponding to non-zero value pixels in the mask "
         "image will be unaffected. Others will be set to zero. This image must "
         "have the same dimensions as frames from SOURCE.")
        ;

    return local_opts;
}

void FrameMasker::applyConfiguration(const po::variables_map &vm,
                                     const config::OptionTable &config_table)
{
    // Background image path
    std::string img_path;
    if (oat::config::getValue(vm, config_table, "mask", img_path, true)) {

        roi_mask_ = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);

        if (roi_mask_.data == NULL)
            throw (std::runtime_error("File \"" + img_path + "\" could not be read."));

        mask_set_ = true;
    }
}

void FrameMasker::filter(cv::Mat &frame)
{
    if (mask_set_)
        frame.setTo(0, roi_mask_ == 0);
}

} /* namespace oat */
