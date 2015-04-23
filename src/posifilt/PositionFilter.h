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

#include "../../lib/shmem/SMServer.h"
#include "../../lib/shmem/SMClient.h"
#include "../../lib/shmem/Position.h"

class PositionFilter {
public:

    PositionFilter(std::string position_source_name, std::string position_sink_name) :
    name(position_sink_name)
    , position_source(position_source_name)
    , position_sink(position_sink_name) {

        position_source.findSharedObject();
    }

    // Position Filters must be able to grab the current position from
    // a position source (such as a Detector)
    virtual void grabPosition(void) = 0;

    // Position Filters must be able filter the position 
    virtual void filterPosition(void) = 0;

    // Position filters must be able serve the filtered position
    virtual void serveFilteredPosition(void) = 0;
    
    // Position filters must be configurable via file
    virtual void configure(std::string config_file, std::string config_key) = 0;

    void stop(void) {
        position_source.notifySelf();
        position_sink.set_running(false);
    }

protected:

    std::string name;
    shmem::SMClient<shmem::Position> position_source;
    shmem::SMServer<shmem::Position> position_sink;
};

#endif	/* POSITIONFILTER_H */

