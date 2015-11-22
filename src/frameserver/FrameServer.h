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

#include <atomic>
#include <opencv2/opencv.hpp>

#include "../../lib/datatypes/Frame.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/SharedCVMat.h"

namespace oat {

/**
 * Abstract base class to be implemented by any FrameServer
 * @param frame_sink_address Address of node to publish shared frames to.
 */
class FrameServer {
public:

    FrameServer(const std::string &frame_sink_address) :
      name_("frameserve[" + frame_sink_address + "]")
    , frame_sink_address_(frame_sink_address)
    {
        // Nothing
    }

    // Oat components are not copyable
    FrameServer(const FrameServer& server) = delete;

    virtual ~FrameServer() {};

    /**
     * FrameServers must be able to connect to a Node that exists in shared memory
     */
    virtual void connectToNode(void) = 0;

    /**
     * FrameServers must be able to serve cv::Mat frames.
     * @return running state. true = stream EOF . false = stream not exhausted.
     */
    virtual bool serveFrame(void) = 0;

    // FrameServers must be configurable via file
    virtual void configure(void) = 0;
    virtual void configure(const std::string &file_name, const std::string &key) = 0;

    // Accessors
    std::string name() const { return name_; }

protected:

    // Component name
    std::string name_;

    // Cameras have a region of interest to crop images
    bool use_roi_ {false};
    cv::Rect_<size_t> region_of_interest_;

    // Frame sink
    const std::string frame_sink_address_;
    oat::Sink<oat::SharedCVMat> frame_sink_;

    // Currently acquired, shared frame
    bool frame_empty_;
    oat::Frame shared_frame_;
};

}       /* namespace oat */
#endif	/* OAT_FRAMESERVER_H */

