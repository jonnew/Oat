//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
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

#ifndef DETECTOR_H
#define	DETECTOR_H

#include <string>
#include <opencv2/opencv.hpp>
#include <boost/thread/mutex.hpp>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/SMServer.h"
#include "../../lib/shmem/SyncSharedMemoryObject.h"

/**
 * Abstract base class to be implemented by any object detector within the 
 * Simple Tracker project. Detector2D's are defined as classes used to identify
 * the _2D_ position of an object within images provided by a frame SOURCE and to
 * publish these detected _2D_ positions to SINK.
 * @param image_source_name Image SOURCE name
 * @param position_sink_name Position SINK name
 */
class Detector2D {
public:

    Detector2D(const std::string& image_source_name, const std::string& position_sink_name) :
      name("posidet[" + image_source_name + "->" + position_sink_name + "]")
    , frame_source(image_source_name)
    , position_sink(position_sink_name)
    , tuning_on(false) {
    }

    // Detector must be able to find an object
    bool process(void) {

        // If we are able to get a an image
        if (frame_source.getSharedMat(current_frame)) {

            position_sink.pushObject(detectPosition(current_frame), 
                                     frame_source.get_current_sample_number());
        }
        
        // If server state is END, return true
        return (frame_source.getSourceRunState() == oat::ServerRunState::END);
    }

    // Detectors must be configurable via file
    virtual void configure(const std::string& config_file, const std::string& config_key) = 0;

    // Accessible from UI thread
    std::string get_name(void) const { return name; }
    void set_tune_mode(bool value) { tuning_on = value; }
    bool get_tune_mode(void) const { return tuning_on; }

protected:
    
    // To be implemented by derived classes
    virtual oat::Position2D detectPosition(cv::Mat& frame_in) = 0;

    // Detectors can allow manual tuning of parameters using this flag
    std::atomic<bool> tuning_on; // This is a shared resource and must be synchronized
 
private:
    
    // Detector name
    std::string name;
    
    // Current frame
    cv::Mat current_frame;

    // The image source (Client side)
    oat::MatClient frame_source;

    // The detected object position destination (Server side)
    oat::SMServer<oat::Position2D> position_sink;
    
};

#endif	/* DETECTOR_H */

