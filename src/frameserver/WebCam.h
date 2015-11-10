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

#include <string>
#include <opencv2/core/mat.hpp>

#include "FrameServer.h"

// Forward declaration
namespace cv {
    class VideoCapture;
}

class WebCam : public FrameServer {

public:

    WebCam(const std::string &frame_sink_address_);

    // Implement FrameServer interface
    void configure(void) override;
    void configure(const std::string &config_file, const std::string &config_key) override;
    void connectToNode(void) override;
    bool serveFrame(void) override;

    // Constants
    static constexpr int64_t MIN_INDEX {0};

private:

    bool aquisition_started_;

    // The webcam object
    int64_t index_;
    std::unique_ptr<cv::VideoCapture> cv_camera_;

};
#endif /* OAT_WEBCAM_H */
