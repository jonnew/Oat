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

#include <vector>

#include "MeanPosition2D.h"

MeanPosition2D::MeanPosition2D(std::vector<std::string> position_source_names, std::string sink_name) :
  PositionCombiner(position_source_names, sink_name) { }

void MeanPosition2D::combineAndServePosition() {

    // Get the current image
    while (client_idx < position_sources.size()) {
        
        if (!position_sources[client_idx]->getSharedObject(*source_positions[client_idx])) {
            return;
        }
        client_idx++;
    }

    client_idx = 0;
    combinePositions();

    position_sink.pushObject(combined_position);
}

/**
 * This function calculates the geometric mean of all positions.
 */
void MeanPosition2D::combinePositions() {

    bool all_positions_valid = true;
    double denominator = 1.0/(double) source_positions.size();
    combined_position.position = datatypes::Point2D(0,0);
    combined_position.velocity = datatypes::Velocity2D(0,0);
    combined_position.head_direction = datatypes::UnitVector2D(0,0);
    
    // Averaging operation
    for (auto pos : source_positions) {
        combined_position.position_valid = 
                all_positions_valid && pos->position_valid;
        combined_position.velocity += 
                denominator * pos->velocity;
        combined_position.head_direction += 
                denominator * pos->head_direction;
    }
}