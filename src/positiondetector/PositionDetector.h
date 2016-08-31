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

#include <boost/program_options.hpp>

#include "../../lib/datatypes/Frame.h"
#include "../../lib/datatypes/Position2D.h"
#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"

namespace po = boost::program_options;

namespace oat {

// Forward decl.
class SharedFrameHeader;

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
                     const std::string &position_sink_address);

    virtual ~PositionDetector() { }

    /**
     * @brief Append type-specific program options.
     * @param opts Program option description to be specialized.
     */
    virtual void appendOptions(po::options_description &opts);

    /**
     * @brief Configure component parameters.
     * @param vm Previously parsed program option value map.
     */
    virtual void configure(const po::variables_map &vm) = 0;

    /**
     * PositionDetectors must be able to connect to a Source and Sink
     * Nodes in shared memory
     */
    virtual void connectToNode(void);

    /**
     * Obtain frame from SOURCE. Detect object position within the frame. Publish
     * detected position to SINK.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    virtual bool process(void);

    // Accessors
    std::string name(void) const { return name_; }

protected:

    /**
     * Perform object position detection.
     * @param Frame to look for object within.
     * @param position Detected object position.
     */
    virtual void detectPosition(cv::Mat &frame, oat::Position2D &position) = 0;
    
    // Detector name
    const std::string name_;

    // List of allowed configuration options    
    std::vector<std::string> config_keys_;

private:

    // Current frame
    oat::Frame internal_frame_;
    oat::Position2D internal_position_ {"internal"};
    oat::Position2D * shared_position_;

    // Frame source
    const std::string frame_source_address_;
    oat::Source<oat::Frame> frame_source_;

    // Position sink
    const std::string position_sink_address_;
    oat::Sink<oat::Position2D> position_sink_;

};

}      /* namespace oat */
#endif /* OAT_POSITIONDETECTOR_H */

