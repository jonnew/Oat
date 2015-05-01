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

/**
 * Abstract base class to be implemented by any object detector within the 
 * Simple Tracker project. Detector2D's are defined as classes used to indentify
 * the _2D_ position of an object within images provided by a frame SOURCE and to
 * publish these detected _2D_ positions to SINK.
 * @param image_source_name Image SOURCE name
 * @param position_sink_name Position SINK name
 */
class Detector2D {
public:

    Detector2D(std::string image_source_name, std::string position_sink_name) :
    image_source(image_source_name)
    , position_sink(position_sink_name)
    , tuning_image_title(position_sink_name + "_tuning")
    , slider_title(position_sink_name + "_sliders")
    , tuning_windows_created(false)
    , tuning_on(false) {

        image_source.findSharedMat();
    }

    // Detector must be able to find an object
    virtual void findObjectAndServePosition(void) = 0;

    // Detectors must be configurable via file
    virtual void configure(std::string file_name, std::string config_key) = 0;

    void set_tune_mode(bool value) {
        tuning_mutex.lock();
        tuning_on = value;
        tuning_mutex.unlock();
    }

    bool get_tune_mode(void) {
        tuning_mutex.lock();
        return tuning_on;
        tuning_mutex.unlock();
    }

    // Detectors must be interruptable

    void stop(void) {
        position_sink.set_running(false);
    }

protected:

    // Detector must implement method  sifting a threshold image to find objects
    virtual void siftBlobs(void) = 0;

    // Detectors must allow manual tuning
    boost::mutex tuning_mutex; // Sync IO and processing thread, which can both manipulate the tuning state
    bool tuning_on; // This is a shared resource and must be synchronized
    bool tuning_windows_created;
    const std::string tuning_image_title, slider_title;
    cv::Mat tune_image;
    virtual void tune(void) = 0;
    virtual void createTuningWindows(void) = 0;

    void addHomography() {
        if (image_source.is_shared_object_found() &&
            (object_position.homography_valid = image_source.is_homography_valid())) {
            object_position.homography = image_source.get_homography();
        }
    }

    // The detected object position
    datatypes::Position2D object_position;

    // The image source (Client side)
    MatClient image_source;

    // The detected object position destination (Server side)
    shmem::SMServer<datatypes::Position2D> position_sink;

};

#endif	/* DETECTOR_H */

