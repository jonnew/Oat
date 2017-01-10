//******************************************************************************
//* File:   UCLAMiniscope.h
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

#ifndef OAT_UCLAMINISCOPE_H
#define OAT_UCLAMINISCOPE_H

#include "FrameServer.h"

#include <chrono>
#include <string>

#include <opencv2/videoio.hpp>

namespace oat {

class UCLAMiniscope : public FrameServer {
public:
    /**
     * @brief Serve test frames from a UCLA miniscope. This is a normal webcam
     * with some extra config stuff hacked into unused control registers (e.g.
     * for controlling excitation LED brightness).
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

    // Camera
    int index_ {0};
    std::unique_ptr<cv::VideoCapture> camera_;
    void setLEDIntensity(size_t index, double value);

    // Frame generation clock
    bool first_frame_ {true};
    std::chrono::steady_clock clock_;
    std::chrono::steady_clock::time_point start_;
};

}      /* namespace oat */
#endif /* OAT_UCLAMINISCOPE_H */
