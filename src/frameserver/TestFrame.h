//******************************************************************************
//* File:   TestFrame.h
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

#ifndef OAT_TESTFRAME_H
#define	OAT_TESTFRAME_H

#include "FrameServer.h"

#include <chrono>
#include <limits>
#include <string>

namespace oat {

class TestFrame : public FrameServer {
public:

    /**
     * @brief Serve test frames using a static image.
     * @param sink_address frame sink address
     */
    explicit TestFrame(const std::string &sink_address);

    void appendOptions(po::options_description &opts) override;
    void configure(const po::variables_map &vm) override;

    void connectToNode(void) override;
    bool process(void) override;

private:

    // Image file
    std::string file_name_;

    // Frame speed
    double frames_per_second_;
    void calculateFramePeriod(void);

    // frame generation clock
    std::chrono::high_resolution_clock clock_;
    std::chrono::duration<double> frame_period_in_sec_;
    std::chrono::high_resolution_clock::time_point tick_;

    // Sample count specification
    uint64_t num_samples_ {std::numeric_limits<int64_t>::max()};

    // Color switch
    oat::PixelColor color_ {oat::PIX_BGR};
};

}       /* namespace oat */
#endif	/* OAT_TESTFRAME_H */
