//******************************************************************************
//* File:   FrameFilter.h
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

#ifndef OAT_FRAMEFILT_H
#define	OAT_FRAMEFILT_H

#include <string>
#include <boost/program_options.hpp>

#include "../../lib/datatypes/Frame.h"
#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"

namespace po = boost::program_options;

namespace oat {

// Forward decl.
class SharedFrameHeader;

/**
 * @brief Abstract frame filter.
 * All concrete frame filter types implement this ABC.
 */
class FrameFilter {
public:

    /**
     * @brief Abstract frame filter.
     * All concrete frame filter types implement this ABC.
     * @param frame_source_address Frame SOURCE node address
     * @param frame_sink_address Frame SINK node address
     */
    FrameFilter(const std::string &frame_source_address,
                const std::string &frame_sink_address);

    virtual ~FrameFilter() { };

    /**
     * @brief FrameServers must be able to connect to a Source and Sink Nodes
     * in shared memory
     */
    virtual void connectToNode(void);

    /**
     * @breif Obtain raw frame from SOURCE. Apply filter function to raw frame. Publish
     * filtered frame to SINK.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    bool processFrame(void);


    virtual void appendOptions(po::options_description &opts); 

    /** 
     * @brief Configure filter parameters.
     * @param vm Previously parsed program option value map.
     */
    virtual void configure(const po::variables_map &vm) { };//= 0;

    ///**
    // * @brief Configure filter parameters.
    // * @param config_file configuration file path
    // * @param config_key configuration key
    // */
    virtual void configure(const std::string &config_file,
                           const std::string &config_key) = 0;

    ///**
    // * @brief Get internal component options.
    // * @return Copy of internal component options.
    // */
    //po::options_description
    //component_options() const { return component_options_; }

    /**
     * @breif Get frame filter name
     * @return name
     */
    std::string name(void) const { return name_; }

    /**
     * @brief Get parameters of frames being processed by this component
     * @return Frame parameters
     */
    oat::Source<oat::SharedFrameHeader>::ConnectionParameters
    frame_parameters(void) const { return frame_parameters_; }

protected:

    /**
     * Perform frame filtering.
     * @param frame to be filtered
     */
    virtual void filter(cv::Mat& frame) = 0;

    // List of allowed configuration options, including those 
    // specified only via config file
    static const char *ALL_CONFIG_OPTS;
po::options_description component_options_;
private:

    // Filter name
    const std::string name_;

    // Frame source
    const std::string frame_source_address_;
    oat::Source<oat::SharedFrameHeader> frame_source_;

    // Frame sink
    const std::string frame_sink_address_;
    oat::Sink<oat::SharedFrameHeader> frame_sink_;

    // Currently processed frame
    oat::Frame internal_frame_;

    // Currently acquired, shared frame
    oat::Frame shared_frame_;
    oat::Source<oat::SharedFrameHeader>::ConnectionParameters frame_parameters_;
};

}      /* namespace oat */
#endif /* OAT_FRAMEFILT_H */
