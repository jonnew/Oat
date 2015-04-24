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

#include "PositionCombiner.h"

PositionCombiner::PositionCombiner(std::string antierior_source_name,
        std::string posterior_source_name,
        std::string sink_name) :
name(sink_name)
, anterior_source(antierior_source_name)
, posterior_source(posterior_source_name)
, position_sink(sink_name)
, current_processing_stage(0) {

    anterior_source.findSharedObject();
    posterior_source.findSharedObject();

}

void PositionCombiner::combineAndServePosition() {

    // Get the current image
    switch (current_processing_stage) {
        case 0:

            if (!anterior_source.getSharedObject(anterior)) {
                return;
            }

            // Fall through
            current_processing_stage = 1;

        case 1:

            if (!posterior_source.getSharedObject(posterior)) {
                return;
            }

            // Fall through
            current_processing_stage = 0;
    }

    calculateGeometricMean();

    position_sink.pushObject(processed_position);
}

void PositionCombiner::calculateGeometricMean() {

    bool both_positions_valid = true;


    if (anterior.position_valid) {

        processed_position.anterior_valid = true;
        processed_position.anterior = anterior.position;
    } else {
        processed_position.position_valid = false;
        processed_position.head_direction_valid = false;
        both_positions_valid = false;
    }

    if (posterior.position_valid) {

        processed_position.posterior_valid = true;
        processed_position.posterior = posterior.position;
    } else {
        processed_position.position_valid = false;
        processed_position.head_direction_valid = false;
        both_positions_valid = false;
    }

    if (both_positions_valid) {
        // If both position measures are valid, take their geometric average
        processed_position.position_valid = true;
        processed_position.position = 0.5 * (anterior.position + posterior.position);
        processed_position.head_direction_valid = true;
        processed_position.head_direction = (anterior.position - posterior.position)*(1.0 / cv::norm(anterior.position - posterior.position));
    }

}