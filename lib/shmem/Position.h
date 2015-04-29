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

#ifndef POSITION_H
#define	POSITION_H

#include <opencv2/core/mat.hpp>

namespace shmem {
    
    typedef cv::Point3f Position3D;
    typedef cv::Point3f Velocity3D;
    typedef cv::Point3f UnitVector3D;

    struct Position {
        
        // Used to get world coordinates from image
        // TODO: Replace with homography transformation matrix
        bool world_coords_valid = false;
        cv::Point3f xyz_origin_in_px;
        float worldunits_per_px_x;
        float worldunits_per_px_y;
        float worldunits_per_px_z;
        
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
        
        Position3D convertPositionToWorldCoords(Position3D position) {
            
            Position3D world_position;
            
            world_position.x = (position.x - xyz_origin_in_px.x) * worldunits_per_px_x;
            world_position.y = (position.y - xyz_origin_in_px.y) * worldunits_per_px_y;
            world_position.z = (position.z - xyz_origin_in_px.z) * worldunits_per_px_z;
            
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