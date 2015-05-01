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

#ifndef POSITION2D_H
#define	POSITION2D_H

#include <opencv2/core/mat.hpp>

namespace shmem {
    
    enum {PIXELS=0, WORLD=1};
    typedef cv::Point3f Position3D;
    typedef cv::Point3f Velocity3D;
    typedef cv::Point3f UnitVector3D;

    struct Position {
        
        // Unless manually changed, we are using pixels as our unit of measure
        int coord_system = PIXELS;
        
        // Used to get world coordinates from image
        bool homography_valid = false;
        cv::Matx33f homography;  
        
        bool position_valid = false;
        Position3D position;
 
        bool anterior_valid = false;
        Position3D anterior;
        
        bool posterior_valid = false;
        Position3D posterior;

        bool velocity_valid = false;
        Velocity3D velocity; 

        bool head_direction_valid = false;
        UnitVector3D head_direction; 
        
        Position3D convertToWorldCoords(const Position3D& position) {
            
            if (coord_system == PIXELS && homography_valid) {
                Position3D world_position; 
            cv::perspectiveTransform(position, world_position, homography); 
            } else {
                
            }
            return world_position;
        }
        
        Position3D convertVelocityToWorldCoords(Velocity3D velocity) {
            
            Position3D world_velocity;
            
            world_velocity.x = velocity.x * worldunits_per_px_x;
            world_velocity.y = velocity.y * worldunits_per_px_y;
            world_velocity.z = velocity.z * worldunits_per_px_z;
            
            return world_velocity;
        }
    };
}

#endif	/* POSITION_H */