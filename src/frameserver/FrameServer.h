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

#include "../../lib/datatypes/Frame.h"
#include "../../lib/shmemdf/Sink.h"

namespace po = boost::program_options;

namespace oat {

/**
 * Abstract base class to be implemented by any FrameServer
 * @param frame_sink_address Address of node to publish shared frames to.
 */
class FrameServer {
public:

    /**
     * @brief Abstract frame server
     * @param sink_address frame sink address
     */
    explicit FrameServer(const std::string &sink_address);
    virtual ~FrameServer() { };

    /**
     * @brief Connect to shared memory node.
     */
    virtual void connectToNode(void) = 0;

    /**
     * FrameServers must be able to serve cv::Mat frames.
     * @return running state. true = stream EOF . false = stream not exhausted.
     */
    virtual bool process(void) = 0;

    /**
     * @brief Append type-specific program options.
     * @param opts Program option description to be specialized.
     */
    virtual void appendOptions(po::options_description &opts);

    /**
     * @brief Configure frame server parameters.
     * @param vm Previously parsed program option value map.
     */
    virtual void configure(const po::variables_map &vm) = 0;

    // Accessors
    std::string name() const { return name_; }

protected:

    // Component name
    std::string name_;

    // List of allowed configuration options, including those
    // specified only via config file
    std::vector<std::string> config_keys_;

    // Cameras can have a region of interest to crop incoming frames
    bool use_roi_ {false};
    cv::Rect_<size_t> region_of_interest_;

    // Frame sink
    const std::string frame_sink_address_;
    oat::Sink<oat::Frame> frame_sink_;

    // Currently acquired, shared frame
    //bool frame_empty_ {true};
    oat::Frame shared_frame_;

    // Internal sample number
    oat::Sample internal_sample_;
};

}       /* namespace oat */
#endif	/* OAT_FRAMESERVER_H */
