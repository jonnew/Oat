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

#include <cstring>
#include <string>
#include <opencv2/core/mat.hpp>

#ifdef OAT_USE_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#include "../../experiments/lib/Source.h"
#include "../../experiments/lib/Sink.h"
#include "../../experiments/lib/SharedCVMat.h"

/**
 * Abstract frame filter.
 * All concrete frame filter types implement this ABC.
 */
class FrameFilter {
public:

    /**
     * Abstract frame filter.
     * All concrete frame filter types implement this ABC.
     * @param frame_source_address Frame SOURCE node address
     * @param frame_sink_address Frame SINK node address
     */
    FrameFilter(const std::string &frame_source_address,
                const std::string &frame_sink_address) :
      name_("framefilt[" + frame_source_address + "->" + frame_sink_address + "]")
    , frame_source_address_(frame_source_address)
    , frame_sink_address_(frame_sink_address)
    {
        // Nothing
    }

    virtual ~FrameFilter() { }

    /**
     * FrameServers must be able to connect to a Source and Sink
     * Nodes in shared memory
     */
    virtual void connectToNode() {

        // Connect to source node and retrieve cv::Mat parameters
        frame_source_.connect(frame_source_address_);
        oat::Source<oat::SharedCVMat>::MatParameters param =
                frame_source_.parameters();

        // Bind to sink sink node and create a shared cv::Mat
        frame_sink_.bind(frame_sink_address_, param.bytes);
        shared_frame_ = frame_sink_.retrieve(param.rows, param.cols, param.type);
    }

    /**
     * Obtain raw frame from SOURCE. Apply filter function to raw frame. Publish
     * filtered frame to SINK.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    virtual bool processFrame(void) {

        // START CRITICAL SECTION //
        ////////////////////////////

        // Wait for sink to write to node
        node_state_ = frame_source_.wait();

        // Clone the shared frame
        internal_frame_ = frame_source_.clone();

        // Tell sink it can continue
        frame_source_.post();

        ////////////////////////////
        //  END CRITICAL SECTION  //

        // Mess with internal frame
        filter(internal_frame_);

        // START CRITICAL SECTION //
        ////////////////////////////

        // Wait for sources to read
        frame_sink_.wait();

        // TODO: For some filters, it may be best for the filter operation to go
        //       here instead of in a non-critical section followed by a copy
        //       operation.
        memcpy(shared_frame_.data, internal_frame_.data,
                internal_frame_.total() * internal_frame_.elemSize());

        // Tell sources there is new data
        frame_sink_.post();

        ////////////////////////////
        //  END CRITICAL SECTION  //

        return (node_state_ == oat::NodeState::END);
    }

    /**
     * Configure filter parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    virtual void configure(const std::string& config_file, const std::string& config_key) = 0;

    /**
     * Get frame filter name
     * @return name
     */
    std::string name(void) const { return name_; }

protected:

    /**
     * Perform frame filtering.
     * @param frame to be filtered
     */
    virtual void filter(cv::Mat& frame) = 0;

private:

    // Filter name.
    const std::string name_;

    // Currently processed frame
    cv::Mat internal_frame_;

    // Frame source
    const std::string frame_source_address_;
    oat::NodeState node_state_;
    oat::Source<oat::SharedCVMat> frame_source_;

    // Frame sink
    const std::string frame_sink_address_;
    oat::Sink<oat::SharedCVMat> frame_sink_;

    // Currently acquired, shared frame
    bool frame_empty_;
    cv::Mat shared_frame_;
};

#endif	/* OAT_FRAMEFILT_H */

