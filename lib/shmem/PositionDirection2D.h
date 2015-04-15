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

#ifndef POSITIONDIRECTION2D_H
#define	POSITIONDIRECTION2D_H

#include "SyncSharedMemoryObject.h"

namespace shmem {

    template<class PointType>
    class PositionDirection2D : public SyncSharedMemoryObject {
        
    public:

        /**
         * Update the 2D position and direction. value is some type of cv::Point (e.g. cv::Pointd,
         * or cv::Pointi). 
         */
        void set_value(PointType& position, double direction_in_radians) { 
            xy = position;
            direction_in_rad = direction_in_radians;
        }

    private:
        PointType xy; 
        double direction_in_rad;
    };
}

#endif	/* POSITIONDIRECTION2D_H */

