//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
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

#ifndef OAT_POSITION3D_H
#define	OAT_POSITION3D_H

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "Position.h"

namespace oat {
    
typedef cv::Point3d Point3D;
typedef cv::Point3d Velocity3D;
typedef cv::Point3d UnitVector3D;

struct Position3D : public Position {
    
    // Unless manually changed, we are using pixels as our unit of measure
    int coord_system = PIXELS;
    
    // Used to get world coordinates from image
    bool homography_valid = false;
    cv::Matx44d homography;  
    
    bool position_valid = false;
    Point3D position;

    bool velocity_valid = false;
    Velocity3D velocity; 

    bool head_direction_valid = false;
    UnitVector3D head_direction; 
    
    Position3D convertToWorldCoordinates() {
        
        if (coord_system == PIXELS && homography_valid) {
            Position2D world_position = *this; 
            
            // Position transforms
            std::vector<Point3D> in_positions;
            std::vector<Point3D> out_positions;
            in_positions.push_back(position);
            // TODO: 3D Transform??
            //cv::perspectiveTransform(in_positions, out_positions, homography);
            //world_position.position = out_positions[0];
            
            // Velocity transform
            std::vector<Velocity3D> in_velocities;
            std::vector<Velocity3D> out_velocities;
            cv::Matx44d vel_homo = homography;
            vel_homo(0,2) = 0.0; // offsets to not apply to velocity
            vel_homo(1,2) = 0.0; // offsets to not apply to velocity
            in_velocities.push_back(velocity);
            
            // TODO: 3D Transform??
            //cv::perspectiveTransform(in_velocities, out_velocities, vel_homo);
            //world_position.velocity = out_velocities[0];
            
            // Head direction is normalized and unit-free, and therefore
            // does not require conversion
            
            // Return value uses world coordinates
            world_position.coord_system = WORLD;
            
            return world_position;
            
        } else {
            return *this;
        }
    }  
};

}      /* namespace datatypes */
#endif /* OAT_POSITION3D_H */
