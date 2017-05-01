//******************************************************************************
//* File:   FileReader.h
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

#ifndef OAT_FILEREADER_H
#define	OAT_FILEREADER_H

#include <chrono>
#include <limits>
#include <string>

#include <opencv2/videoio.hpp>

#include "FrameServer.h"

namespace oat {

class FileReader : public FrameServer {
public:

    explicit FileReader(const std::string &sink_name);

private:
    // Component Interface
    bool connectToNode(void) override;
    int process(void) override;

    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    // Video file
    // TODO: This thing kinda sucks. Should use ffmpeg directly, e.g. to read
    // mono frames without having to convert
    cv::VideoCapture file_reader_;

    // Frame read clock
    oat::Token::Seconds read_period_;
    std::chrono::high_resolution_clock clock_;
    std::chrono::high_resolution_clock::time_point tick_;

    // Frames to skip between reads
    int skip_{0};

    // Video boundaries
    std::vector<int> bounds_{{0, -1}};

    // Region of interest
    bool use_roi_{false};
    cv::Rect_<size_t> region_of_interest_;
};

}       /* namespace oat */
#endif	/* OAT_FILEREADER_H */

