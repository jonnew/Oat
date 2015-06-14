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
#include <cmath>

#include "../../lib/shmem/Signals.h"
#include "MeanPosition2D.h"

MeanPosition2D::MeanPosition2D(std::vector<std::string> position_source_names, std::string sink_name) :
  PositionCombiner(position_source_names, sink_name) { }


/**
 * This function calculates the geometric mean of all source positions.
 * @param sources Source positions
 * @return Combined position
 */
oat::Position2D MeanPosition2D::combinePositions(const std::vector<oat::Position2D*>& sources) {

    double mean_denom = 1.0/(double) sources.size(); 
    oat::Position2D combined_position;
    combined_position.position = oat::Point2D(0,0);
    combined_position.velocity = oat::Velocity2D(0,0);
    combined_position.head_direction = oat::UnitVector2D(0,0);
    
    // Averaging operation
    for (auto pos : sources) {
        
        // Position
        if (pos->position_valid)
            combined_position.position += mean_denom * pos->position;
        else
            combined_position.position_valid = false;
        
        // Velocity
        if (pos->velocity_valid)
            combined_position.velocity += mean_denom * pos->velocity;
        else
            combined_position.velocity_valid = false;
        
        // Head direction
        if (pos->head_direction_valid)
            combined_position.head_direction += pos->head_direction;
        else
           combined_position.head_direction_valid = false; 
        
    }
    
    // Renormalize head-direction unit vector
    if (combined_position.head_direction_valid)
    {
        double mag = std::sqrt(
                std::pow(combined_position.head_direction.x, 2.0) +
                std::pow(combined_position.head_direction.y, 2.0));

        combined_position.head_direction = 
                combined_position.head_direction/mag;
    }
    
    return combined_position;
}