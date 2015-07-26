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

#ifndef POSITIONDETECTOR_H
#define	POSITIONDETECTOR_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/SMServer.h"

namespace oat {
    class Position2D;
}

/**
 * Abstract object position detector.
 * All concrete object position detector types implement this ABC.
 */
class PositionDetector {
public:

    /**
     * Abstract object position detector.
     * All concrete object position detector types implement this ABC.
     * @param image_source_name Frame SOURCE name
     * @param position_sink_name Position SINK name
     */
    PositionDetector(const std::string& image_source_name, const std::string& position_sink_name) :
      name("posidet[" + image_source_name + "->" + position_sink_name + "]")
    , frame_source(image_source_name)
    , position_sink(position_sink_name) {
    }

    virtual ~PositionDetector() { }

    /**
     * Obtain frame from SOURCE. Detect object position within the frame. Publish
     * detected position to SINK.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    bool process(void) {

        // If we are able to get a an image
        if (frame_source.getSharedMat(current_frame)) {

            position_sink.pushObject(detectPosition(current_frame), 
                                     frame_source.get_current_sample_number());
        }
        
        // If server state is END, return true
        return (frame_source.getSourceRunState() == oat::ServerRunState::END);
    }

    /**
     * Configure filter parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    virtual void configure(const std::string& config_file, const std::string& config_key) = 0;

    /**
     * Get FrameFilter name
     * @return name 
     */
    std::string get_name(void) const { return name; }

protected:
    
    /**
     * Perform object position detection.
     * @param frame frame to look for object in.
     * @return detected object position.
     */
    virtual oat::Position2D detectPosition(cv::Mat& frame) = 0;
    
    // Detector name
    const std::string name;
 
private:

    // Current frame
    cv::Mat current_frame;

    // Frame SOURCE object for receiving frames
    oat::MatClient frame_source;

    // Position SINK object for publishing detected positions
    oat::SMServer<oat::Position2D> position_sink;
};

#endif	/* POSITIONDETECTOR_H */

