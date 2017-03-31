//******************************************************************************
//* File:   WebCam.h
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

#ifndef OAT_WEBCAM_H
#define OAT_WEBCAM_H

#include "FrameServer.h"

#include <chrono>
#include <memory>
#include <string>

#include <opencv2/videoio.hpp>

namespace oat {

class WebCam : public FrameServer {
public:
    /**
     * @brief Serve test frames from a webcam.
     * @param sink_address frame sink address
     */
    using FrameServer::FrameServer;

private:
    // Component Interface
    bool connectToNode(void) override;
    int process(void) override;

    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    // OpenCV video capture wrapper for v4l2
    std::unique_ptr<cv::VideoCapture> camera_;

    // frame generation clock
    bool first_frame_{true};
    std::chrono::steady_clock clock_;
    std::chrono::steady_clock::time_point start_;

    // WebCam can have a region of interest to crop incoming frames
    bool use_roi_{false};
    cv::Rect_<size_t> region_of_interest_;

    // In the case that we are using camera for frame rate, detect if the FPS
    // is bogus
    bool variable_fps_{false};
};

}      /* namespace oat */
#endif /* OAT_WEBCAM_H */
