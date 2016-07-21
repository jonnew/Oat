//******************************************************************************
//* File:   MaskSubtractor.cpp
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
#include <vector>
#include <cmath>
#include <cpptoml.h>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"

#include "MaskSubtractor.h"

namespace oat {

MaskSubtractor::MaskSubtractor(const std::vector<std::string> &frame_source_addresses,
                               const std::string &frame_sink_address) :
  FrameCombiner(frame_source_addresses, frame_sink_address)
{
    // Nothing
}

void MaskSubtractor::configure(const std::string& config_file, const std::string& config_key) {

    // Available options
    const std::vector<std::string> options {"processed-frame"};

    // This will throw cpptoml::parse_exception if a file
    // with invalid TOML is provided
    auto config = cpptoml::parse_file(config_file);

    // See if a configuration was provided
    if (config->contains(config_key)) {

        // Get this components configuration table
        auto this_config = config->get_table(config_key);

        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);

        // Heading anchor
        if (oat::config::getValue(this_config, "masked-frame",
                masked_frame_idx_,
                static_cast<int64_t>(0),
                static_cast<int64_t>(num_sources() - 1))) {
        }

    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

void MaskSubtractor::combine(const std::vector<oat::Frame> &source_frames,
                             oat::Frame &combined_frame) {

    // Generate Mask
    cv::Mat mask = cv::Mat::ones(source_frames[0].size(), CV_8UC3);
    for (auto &f : source_frames)
        // TODO: respect mask type
        cv::bitwise_and(f, mask, mask);

    combined_frame.setTo(0, mask == 0);
}

} /* namespace oat */
