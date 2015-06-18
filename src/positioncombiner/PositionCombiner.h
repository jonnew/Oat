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
      name("posicom[" + position_source_names[0] + "...->" + sink_name + "]") 
    , position_sink(sink_name)
    , client_idx(0) {
         
        for (auto &name : position_source_names) {
            
            position_sources.push_back(new oat::SMClient<oat::Position2D>(name));
            source_positions.push_back(new oat::Position2D); 
        }
    }

    // Public 'run' method
    bool process() {

        // Are all sources running?
        bool sources_running = true;
    
        // Get current positions
        while (client_idx < position_sources.size()) {

            // Check if source is sill running
            sources_running &= (position_sources[client_idx]->getSourceRunState()
                    == oat::ServerRunState::RUNNING);

            if (!(position_sources[client_idx]->getSharedObject(*source_positions[client_idx]))) {
                return sources_running;
            }
            
            client_idx++;
        }

        client_idx = 0;
        combined_position = combinePositions(source_positions);

        position_sink.pushObject(combined_position, position_sources[0]->get_current_time_stamp());
        
        return sources_running;

    }

    std::string get_name(void) const { return name; }

protected:

    // All position combiners must be able to combine the position_sources
    // list to provide a single combined position output
    virtual oat::Position2D combinePositions(const std::vector<oat::Position2D*>& sources) = 0;
    
private:
    
    std::string name;

    // For multi-source processing, we need to keep track of all the sources
    // we have finished reading from each processing step
    std::vector<oat::SMClient<oat::Position2D> >::size_type client_idx;

    // Positions to be combined
    std::vector<oat::Position2D* > source_positions;
    std::vector<oat::SMClient<oat::Position2D>* > position_sources;

    // Combined position server
    oat::Position2D combined_position;
    oat::SMServer<oat::Position2D> position_sink;
};

#endif	// POSITIONCOMBINER_H

