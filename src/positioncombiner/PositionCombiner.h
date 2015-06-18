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
#include <boost/dynamic_bitset.hpp>

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
    , position_read_success(position_source_names.size()) {
         
        for (auto &name : position_source_names) {
            
            position_sources.push_back(new oat::SMClient<oat::Position2D>(name));
            source_positions.push_back(new oat::Position2D); 
        }
        
        number_of_sources = position_sources.size();
    }

    ~PositionCombiner() {

        // Release resources
        for (auto &value : position_sources) {
            delete value;
        }

        for (auto &value : source_positions) {
            delete value;
        }
    }

    // Public 'run' method
    bool process() {

        // Are all sources running?
        bool sources_eof = false;

        for (int i = 0; i < position_sources.size(); i++) {

            sources_eof |= (position_sources[i]->getSourceRunState()
                    == oat::ServerRunState::END);

            // Check if we need to read position_client_idx, or if the read has been
            // performed already
            if (position_read_success[i])
                continue;

            position_read_success[i] =
                    position_sources[i]->getSharedObject(*source_positions[i]);
        }

        // If we have not finished reading _any_ of the clients, we cannot proceed
        if (position_read_success.all()) {
            
            // Reset the position client read counter
            position_read_success.reset();

            combined_position = combinePositions(source_positions);

            position_sink.pushObject(combined_position, position_sources[0]->get_current_time_stamp());
        }

        return sources_eof;
    }

    std::string get_name(void) const { return name; }
    
    // Frame filters must be configurable
    virtual void configure(const std::string& config_file, const std::string& config_key) = 0;

protected:

    // All position combiners must be able to combine the position_sources
    // list to provide a single combined position output
    virtual oat::Position2D combinePositions(const std::vector<oat::Position2D*>& sources) = 0;
    
    size_t number_of_sources;
    
private:
    
    std::string name;

    // Positions to be combined
    std::vector<oat::Position2D* > source_positions;
    std::vector<oat::SMClient<oat::Position2D>* > position_sources;
    boost::dynamic_bitset<> position_read_success;

    // Combined position server
    oat::Position2D combined_position;
    oat::SMServer<oat::Position2D> position_sink;
};

#endif	// POSITIONCOMBINER_H

