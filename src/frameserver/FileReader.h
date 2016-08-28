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

    FileReader(const std::string &sink_name);

    void appendOptions(po::options_description &opts) override;
    void configure(const po::variables_map &vm) override;

    void connectToNode(void) override;
    bool process(void) override;

private:

    // Video file
    cv::VideoCapture file_reader_;

    // Playback speed
    double frames_per_second_;
    void calculateFramePeriod(void);

    // Region of interest
    cv::Rect_<size_t> region_of_interest_;

    // Frame generation clock
    std::chrono::high_resolution_clock clock_;
    std::chrono::duration<double> frame_period_in_sec_;
    std::chrono::high_resolution_clock::time_point tick_;
};

}       /* namespace oat */
#endif	/* OAT_FILEREADER_H */

