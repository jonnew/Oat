//******************************************************************************
//* File:   Threshold.h
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

#ifndef OAT_THRESHOLD_H
#define	OAT_THRESHOLD_H

#include "FrameFilter.h"

namespace oat {

/**
 * A frame intensity thresholder
 */
class Threshold : public FrameFilter {
public:

    /**
     * A basic intensity thresholder.
     * @param frame_source_address raw frame source address
     * @param frame_sink_address filtered frame sink address
     */
    Threshold(const std::string &frame_source_address,
              const std::string &frame_sink_address);

    void appendOptions(po::options_description &opts) override;
    void configure(const po::variables_map &vm) override;

private:

    void filter(cv::Mat &frame) override;

    // Intensity threshold boundaries
    int i_min_ {0};
    int i_max_ {256};
};

}      /* namespace oat */
#endif /* OAT_THRESHOLD_H */
