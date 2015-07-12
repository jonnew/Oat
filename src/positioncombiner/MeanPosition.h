//******************************************************************************
//* File:   MeanPosition.h
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

#ifndef MEANPOSITION_H
#define	MEANPOSITION_H

#include "PositionCombiner.h"

/**
 * A mean position combiner.
 */
class MeanPosition : public PositionCombiner {
public:
    
    /**
     * A mean position combiner.
     * A mean position combiner to generate a mean position from 2 or more
     * source positions. Can be used to generate the mean position heading
     * using a specified anchor position as a reference.
     * @param position_source_names
     * @param sink_name
     */
    MeanPosition(std::vector<std::string> position_source_names, std::string sink_name);
    
    void configure(const std::string& config_file, const std::string& config_key);
    
private:

    /**
     * Calculate the mean of SOURCE positions.
     * @param sources SOURCE positions to combine
     * @return mean position
     */
    oat::Position2D combinePositions(const std::vector<oat::Position2D*>& sources);
    
    // Should a heading be generated?
    bool generate_heading;
    
    // SOURCE position to be used ad an anchor when calculating headings
    int64_t heading_anchor_idx;

};

#endif	/* MEANPOSITION2D_H */

