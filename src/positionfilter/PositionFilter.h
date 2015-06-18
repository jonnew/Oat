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
#include "../../lib/datatypes/Position2D.h"

class PositionFilter { 
public:

    PositionFilter(const std::string& position_source_name, const std::string& position_sink_name) :
      name("posifilt[" + position_source_name + "->" + position_sink_name + "]")
    , position_source(position_source_name)
    , position_sink(position_sink_name)
    , tuning_on(false) { }

    virtual ~PositionFilter() { }

    // Execute filtering operation
    bool process(void) {

        if (position_source.getSharedObject(raw_position)) {
            
            position_sink.pushObject(filterPosition(raw_position), 
                                     position_source.get_current_time_stamp());
   
        }
        
        // If server state is END, return true
        return (position_source.getSourceRunState() == oat::ServerRunState::END);  
    }

    // Position filters must be configurable via file
    virtual void configure(const std::string& config_file, const std::string&  config_key) = 0;

    // Accessors
    std::string get_name(void) const { return name; }
    void set_tune_mode(bool value) { tuning_on = value; }
    bool get_tune_mode(void) { return tuning_on; }
    
    //void stop(void) {position_sink.set_running(false); }

    
protected:
    
    // Position Filters must be able filter the position 
    virtual oat::Position2D filterPosition(oat::Position2D& position_in) = 0;
    std::atomic<bool> tuning_on; // This is a shared resource and must be synchronized
        
private:
    
    std::string name;
    oat::SMClient<oat::Position2D> position_source;
    oat::Position2D raw_position;
    oat::SMServer<oat::Position2D> position_sink;
    
};

#endif	/* POSITIONFILTER_H */

