//******************************************************************************
//* File:   FrameServer.h
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
//*****************************************************************************

#ifndef OAT_FRAMESERVER_H
#define	OAT_FRAMESERVER_H

#include <string>

#include <boost/program_options.hpp>
#include <opencv2/core.hpp>

#include "../../lib/base/Component.h"
#include "../../lib/base/Configurable.h"
#include "../../lib/datatypes/Frame.h"
#include "../../lib/shmemdf/Sink.h"

namespace po = boost::program_options;

namespace oat {

class FrameServer : public Component, public Configurable {
public:
    /**
     * @brief Abstract frame server
     * @param frame_sink_address Address of node to publish shared frames to.
     */
    explicit FrameServer(const std::string &sink_address);
    virtual ~FrameServer() { };

    // Component Interface
    oat::ComponentType type(void) const override { return oat::frameserver; };
    std::string name(void) const override { return name_; }

protected:
    // Component name
    std::string name_;

    // Cameras can have a region of interest to crop incoming frames
    bool use_roi_ {false};
    cv::Rect_<size_t> region_of_interest_;

    // Frame sink
    const std::string frame_sink_address_;
    oat::Sink<oat::Frame> frame_sink_;

    // Currently acquired, shared frame
    oat::Frame shared_frame_;
};

}       /* namespace oat */
#endif	/* OAT_FRAMESERVER_H */
