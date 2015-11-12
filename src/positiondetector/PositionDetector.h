//******************************************************************************
//* File:   PositionDetector.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
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
//****************************************************************************

#ifndef OAT_POSITIONDETECTOR_H
#define	OAT_POSITIONDETECTOR_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/datatypes/Position2D.h"
#include "../../experiments/lib/Source.h"
#include "../../experiments/lib/Sink.h"
#include "../../experiments/lib/SharedCVMat.h"

namespace oat {

/**
 * Abstract object position detector.
 * All concrete object position detector types implement this ABC.
 */
class PositionDetector {
public:

    /**
     * Abstract object position detector.
     * All concrete object position detector types implement this ABC.
     * @param frame_source_address Frame SOURCE node address
     * @param position_sink_address Position SINK node address
     */
    PositionDetector(const std::string &frame_source_address,
                     const std::string &position_sink_address) :
      name_("posidet[" + frame_source_address + "->" + position_sink_address + "]")
    , frame_source_address_(frame_source_address)
    , position_sink_address_(position_sink_address)
    {
        // Nothing
    }

    virtual ~PositionDetector() { }

    /**
     * PositionDetectors must be able to connect to a Source and Sink
     * Nodes in shared memory
     */
    virtual void connectToNode() {

        // Connect to source node and retrieve cv::Mat parameters
        frame_source_.connect(frame_source_address_);

        // Bind to sink sink node and create a shared cv::Mat
        position_sink_.bind(position_sink_address_);
        shared_position_ = *position_sink_.retrieve();
    }

    /**
     * Obtain frame from SOURCE. Detect object position within the frame. Publish
     * detected position to SINK.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    bool process(void) {

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
        internal_position_ = detectPosition(internal_frame_);

        // START CRITICAL SECTION //
        ////////////////////////////

        // Wait for sources to read
        position_sink_.wait();

        shared_position_ = internal_position_;

        // Tell sources there is new data
        position_sink_.post();

        ////////////////////////////
        //  END CRITICAL SECTION  //

        return (node_state_ == oat::NodeState::END);
    }

    /**
     * Configure filter parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    virtual void configure(const std::string &config_file, const std::string &config_key) = 0;

    // Accessors
    std::string name(void) const { return name_; }
    virtual void set_tuning(bool value) { tuning_on_ = value; }

protected:

    /**
     * Perform object position detection.
     * @param frame frame to look for object in.
     * @return detected object position.
     */
    virtual oat::Position2D detectPosition(cv::Mat &frame) = 0;

    // Detector name
    const std::string name_;

    // Use GUI to tune detection parameters
    bool tuning_on_ {false};

private:

    // Current frame
    cv::Mat internal_frame_;
    oat::Position2D internal_position_, shared_position_;

    // Frame source
    const std::string frame_source_address_;
    oat::NodeState node_state_;
    oat::Source<oat::SharedCVMat> frame_source_;

    // Position sink
    const std::string position_sink_address_;
    oat::Sink<oat::Position2D> position_sink_;

};

}       /* namespace oat */
#endif	/* OAT_POSITIONDETECTOR_H */

