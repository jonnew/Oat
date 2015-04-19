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

    struct Position {

        bool position_valid = false;
        cv::Point3f position;
        
        bool anterior_valid = false;
        cv::Point3f anterior;
        
        bool posterior_valid = false;
        cv::Point3f posterior;

        bool velocity_valid = false;
        cv::Point3f velocity; 

        bool head_direction_valid = false;
        cv::Point3f head_direction; 
    };
}

#endif	/* POSITION2D_H */