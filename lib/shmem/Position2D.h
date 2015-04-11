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

#include "SyncSharedMemoryObject.h"

namespace shmem {

    class Position2D : public SyncSharedMemoryObject {
        
    public:

        /**
         * Update the 2D position. The absolute (mm) position is updated
         * automatically if mm_per_px has been set.
         * @param xy_px_in xy position in pixels
         */
        void set_value(std::array<int,2> xy_px_in) { 
            xy_px = xy_px_in;
            if (mm_conversion_set) {
                getmmFromPx();
            }
        }
        
        /**
         * Set the mm_per_px conversion value
         * @param value mm per pixel
         */
        void set_mm_per_px(double value) {
            mm_per_px = value;
            mm_conversion_set = true;     
        }
        
    private:
        double mm_per_px = 10.0;
        bool mm_conversion_set = false;
        std::array<int,2> xy_px = {0, 0};
        std::array<double,2> xy_mm = {0.0, 0.0};
        void getmmFromPx(void) { xy_mm = {mm_per_px * xy_px[0], mm_per_px * xy_px[1] }; }     
    };
}

#endif	/* POSITION2D_H */

