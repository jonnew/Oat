//******************************************************************************
//* File:   HomographyTransform2D.h
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

#ifndef HOMGRAPHICTRANSFORM2D_H
#define	HOMGRAPHICTRANSFORM2D_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "PositionFilter.h"

/**
 * A 2D homography.
 */
class HomographyTransform2D : public PositionFilter {
public:
    
    /**
     * A 2D homography.
     * Transform position units. Typically used to perform projective transformation
     * to map pixels coordinates to world coordinates. 
     * @param position_source_name Un-filtered position SOURCE name
     * @param position_sink_name Filtered position SINK name
     */
    HomographyTransform2D(const std::string& position_source_name, const std::string& position_sink_name);

    void configure(const std::string& config_file, const std::string& config_key);

private:

    // 2D homography matrix
    bool homography_valid;
    cv::Matx33d homography;
    
    // Filtered position
    oat::Position2D filtered_position;
    
    /**
     * Apply homography transform.
     * @param position_in Un-projected position SOURCE
     * @return projected position
     */
    oat::Position2D filterPosition(oat::Position2D& position_in);
};

#endif	/* HOMGRAPHICTRANSFORM2D_H */

