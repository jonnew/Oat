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
#ifndef POSITIONFILTER_H
#define	POSITIONFILTER_H

#include <atomic>
#include <boost/thread/mutex.hpp>

#include "../../lib/shmem/SMServer.h"
#include "../../lib/shmem/SMClient.h"
#include "../../lib/datatypes/Position.h"


class PositionFilter {
public:

    PositionFilter(std::string position_source_name, std::string position_sink_name) :
    name(position_sink_name)
    , position_source(position_source_name)
    , position_sink(position_sink_name)
    , canvas_hw(500.0)
    , canvas_border(100.0)
    , tuning_image_title(position_sink_name + "_tuning")
    , slider_title(position_sink_name + "_sliders") {

        position_source.findSharedObject();
    }

    // Execute filtering operation

    void filterPositionAndServe(void) {

        if (grabPosition()) {
            filterPosition();
            serveFilteredPosition();
        }
    }

    // Position filters must be configurable via file
    virtual void configure(std::string config_file, std::string config_key) = 0;

    // Accessors

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
    
    void stop(void) {position_sink.set_running(false); }

protected:
    
    const float canvas_hw;
    const float canvas_border;

    std::string name;
    shmem::SMClient<shmem::Position> position_source;
    shmem::Position raw_position;
    shmem::SMServer<shmem::Position> position_sink;
    shmem::Position filtered_position;

    // tuning on or off
    std::string tuning_image_title, slider_title;
    bool tuning_on; // This is a shared resource and must be synchronized
    boost::mutex tuning_mutex; // Sync IO and processing thread, which can both manipulate the tuning state
    
    // Position Filters must be able to grab the current position from
    // a position source (such as a Detector)
    virtual bool grabPosition(void) = 0;

    // Position Filters must be able filter the position 
    virtual void filterPosition(void) = 0;

    // Position filters must be able serve the filtered position
    virtual void serveFilteredPosition(void) = 0;
    
    // Draw the position on a cv::Mat for tuning purposes
    virtual void drawPosition(cv::Mat& canvas, const datatypes::Position& position) = 0;
};

#endif	/* POSITIONFILTER_H */

