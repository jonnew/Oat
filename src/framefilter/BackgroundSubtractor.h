//******************************************************************************
//* File:   BackgroundSubtractor.h
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

#ifndef OAT_BACKGROUNDSUBTRACTOR_H
#define	OAT_BACKGROUNDSUBTRACTOR_H

#include "FrameFilter.h"

namespace oat {

class BackgroundSubtractor : public FrameFilter {
public:
    /**
     * A basic background subtractor.
     * Subtract a frame image from a frame stream. The background frame is
     * the first frame obtained from the SOURCE frame stream, or can be
     * supplied via configuration file.
     * @param frame_source_address raw frame source address
     * @param frame_sink_address filtered frame sink address
     */
    BackgroundSubtractor(const std::string &frame_souce_address,
                         const std::string &frame_sink_address);

private:
    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;


    // Is the background frame set?
    bool background_set_ {false};

    // The background frame(s)
    cv::Mat background_frame_;
    cv::Mat background_frame_f_;

    // Background update rate
    double alpha_ {0.0};

    /**
     * Apply background subtraction.
     * @param frame unfiltered frame
     * @return filtered frame
     */
    void filter(cv::Mat &frame) override;

    // Set the background frame
    void setBackgroundImage(const cv::Mat&);
};

}      /* namespace oat */
#endif /* OAT_BACKGROUNDSUBTRACTOR_H */
