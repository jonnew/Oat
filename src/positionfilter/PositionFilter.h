//******************************************************************************
//* File:   PositionFilter.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
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
#include "../../lib/datatypes/Position2D.h"

/**
 * Abstract position filter.
 * All concrete position filter types implement this ABC.
 */
class PositionFilter { 
public:

    /**
     * Abstract position filter.
     * All concrete position filter types implement this ABC.
     * @param position_source_name Un-filtered position SOURCE name
     * @param position_sink_name Filtered position SINK name
     */
    PositionFilter(const std::string& position_source_name, const std::string& position_sink_name) :
      name("posifilt[" + position_source_name + "->" + position_sink_name + "]")
    , position_source(position_source_name)
    , position_sink(position_sink_name) { 
      
          position.set_label(position_sink_name);
    }

    virtual ~PositionFilter() { }

    /**
     * Obtain un-filtered position from SOURCE. Filter position. Publish filtered 
     * position to SINK.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    bool process(void) {

        if (position_source.getSharedObject(position)) {
            
            position_sink.pushObject(filterPosition(position), 
                                     position_source.get_current_time_stamp());
   
        }
        
        // If server state is END, return true
        return (position_source.getSourceRunState() == oat::ServerRunState::END);  
    }

    /**
     * Configure position filter parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    virtual void configure(const std::string& config_file, const std::string&  config_key) = 0;

    // Accessors
    std::string get_name(void) const { return name; }

protected:

    /**
     * Perform position filtering.
     * @param position_in Un-filtered position SOURCE
     * @return filtered position
     */
    virtual oat::Position2D filterPosition(oat::Position2D& position_in) = 0;
        
private:
    
    // Filter name
    const std::string name;
    
    // Un-filtered position SOURCE object
    oat::SMClient<oat::Position2D> position_source;
    
    // Un-filtered position
    oat::Position2D position;
    
    // Filtered position SINK object
    oat::SMServer<oat::Position2D> position_sink;
};

#endif	/* POSITIONFILTER_H */

