//******************************************************************************
//* File:   PositionCombiner.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
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
//******************************************************************************

#ifndef POSITIONCOMBINER_H
#define	POSITIONCOMBINER_H

#include <string>
#include <boost/dynamic_bitset.hpp>

#include "../../lib/shmem/SMServer.h"
#include "../../lib/shmem/SMClient.h"
#include "../../lib/datatypes/Position.h"

/**
 * Abstract position combiner.
 * All concrete position combiner types implement this ABC.
 */
class PositionCombiner { 
public:

    /**
     * Abstract position combiner.
     * All concrete position combiner types implement this ABC.
     * @param position_sources A vector of position SOURCE names
     * @param sink Combined position SINK name
     */
    PositionCombiner(std::vector<std::string> position_source_names, std::string sink_name) :
      name("posicom[" + position_source_names[0] + "...->" + sink_name + "]") 
    , position_sink(sink_name)
    , number_of_position_sources(position_source_names.size())
    , position_read_required(number_of_position_sources)
    , sources_eof(false) {
         
        for (auto &name : position_source_names) {
            
            position_sources.push_back(new oat::SMClient<oat::Position2D>(name));
            source_positions.push_back(new oat::Position2D); 
        }
        
        position_read_required.set();
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

    /**
     * Obtain positions from all SOURCES. Combine positions. Publish combined position
     * to SINK.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    bool process() {

        // Make sure all sources are still running
        for (int i = 0; i < number_of_position_sources; i++) {
            
            sources_eof |= (position_sources[i]->getSourceRunState()
                    == oat::SinkState::END);
        }

        boost::dynamic_bitset<>::size_type i = position_read_required.find_first();

        while (i < number_of_position_sources) {

            position_read_required[i] =
                    !position_sources[i]->getSharedObject(*source_positions[i]);

            i = position_read_required.find_next(i);
        }

        // If we have not finished reading _any_ of the clients, we cannot proceed
        if (position_read_required.none()) {

            // Reset the position client read counter
            position_read_required.set();
            combined_position = combinePositions(source_positions);
            position_sink.pushObject(combined_position, position_sources[0]->get_current_time_stamp());
        }

        return sources_eof;
    }

    std::string get_name(void) const { return name; }
    
    /**
     * Configure position combiner parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    virtual void configure(const std::string& config_file, const std::string& config_key) = 0;

protected:

    /**
     * Perform position combination.
     * @param sources SOURCE position servers
     * @return combined position
     */
    virtual oat::Position2D combinePositions(const std::vector<oat::Position2D*>& sources) = 0;
    
    /**
     * Get the number of SOURCE positions.
     * @return number of SOURCE positions
     */
    int get_number_of_sources(void) const {return number_of_position_sources; };
    
private:
    
    // Combiner name
    std::string name;
    
    // Position SOURCES object for un-combined positions
    std::vector<oat::Position2D* > source_positions; // Positions to be combined
    std::vector<oat::SMClient<oat::Position2D>* > position_sources; // Position SOURCES
    boost::dynamic_bitset<>::size_type number_of_position_sources;
    boost::dynamic_bitset<> position_read_required;
    bool sources_eof;

    // Combined position
    oat::Position2D combined_position;
    
    // Position SINK object for publishing combined position
    oat::SMServer<oat::Position2D> position_sink;
};

#endif	// POSITIONCOMBINER_H

