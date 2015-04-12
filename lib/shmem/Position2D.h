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

#include "SyncSharedMemoryObject.h"



namespace shmem {

    class Position2D : public SyncSharedMemoryObject {
        
    public:

        /**
         * Update the 2D position. The absolute (mm) position is updated
         * automatically if mm_per_px has been set.
         * @param xy_px_in xy position in pixels
         */
        void set_value(cv::Point2i& value) { 
            xy_px = value;
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
        
        /**
         * Set the position label for this point (e.g. posterior vs. anterior)
         * @param value 
         */
        void set_position(std::string value) {
            position = value;
        }
        
    private:
        double mm_per_px;
        bool mm_conversion_set = false;
        cv::Point2i xy_px;
        cv::Point2d xy_mm;
        std::string position;
        void getmmFromPx(void) { xy_mm = mm_per_px * xy_px; }     
    };
}

#endif	/* POSITION2D_H */

