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

#ifndef POSITIONCOMBINER_H
#define	POSITIONCOMBINER_H

#include <string>

#include "../../lib/shmem/SMServer.h"
#include "../../lib/shmem/SMClient.h"
#include "../../lib/datatypes/Position.h"

/**
 * Abstract base class to be implemented by any Position Combiner filter within
 * the Simple Tracker project.
 * @param position_sources A vector of position SOURCE names
 * @param sink Combined position SINK name
 */
class PositionCombiner { // TODO: Position2D -> Position somehow
public:

    PositionCombiner(std::vector<std::string> position_source_names, std::string sink_name) :
      name(sink_name)
    , position_sink(sink_name)
    , client_idx(0) {
         
        for (auto &name : position_source_names) {
            
            position_sources.push_back(new shmem::SMClient<datatypes::Position2D>(name));
            source_positions.push_back(new datatypes::Position2D);
            position_sources.back()->findSharedObject();    
        }
    }

    // All position combiners must implement a method to combine positions and
    // publish the result
    virtual void combineAndServePosition(void) = 0;

    std::string get_name(void) { return name; }
    void stop(void) {position_sink.set_running(false); }

protected:

    std::string name;

    // For multi-source processing, we need to keep track of all the sources
    // we have finished reading from each processing step
    std::vector<shmem::SMClient<datatypes::Position2D> >::size_type client_idx;

    // Positions to be combined
    std::vector<datatypes::Position2D* > source_positions;
    std::vector<shmem::SMClient<datatypes::Position2D>* > position_sources;

    // Combined position server
    datatypes::Position2D combined_position;
    shmem::SMServer<datatypes::Position2D> position_sink;

    // All position combiners must be able to combine the position_sources
    // list to provide a single combined position output
    virtual void combinePositions(void) = 0;
};

#endif	// POSITIONCOMBINER_H

