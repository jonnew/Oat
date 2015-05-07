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

// TODO: I would like Positions to have an associated labe, but std::string's
// allocator is not appropriate for shmem. I need to use the boost IP versions
// of the string container, but its quite complicated
//#include <boost/interprocess/allocators/allocator.hpp>
//#include <boost/interprocess/containers/string.hpp>

namespace datatypes {

    enum coordinate_system_type {
        PIXELS = 0, WORLD = 1
    };

    struct Position {

        Position() { }   
        virtual ~Position() = 0;
      
        // Positions use one of two coordinate systems 
        // PIXELS - units are referenced to the sensory array of the digital camera. 
        //          origin in the upper left hand corner.
        // WORLD  - Defined by the homography matrix.
        int coord_system = PIXELS;
        
        // Time keeping
        unsigned int time_stamp; // Time-stamp of this position, respecting buffer overruns
        unsigned int index;      // Order index of this position, disrespecting buffer overruns

        // Positions must be able to convert themselves to world
        // coordinate system if they contain the appropriate
        // homography matracies
        //virtual Position convertToWorldCoordinates(void) = 0;
        // TODO: why can't this base class use the more general Position type instead
        // of Position2D etc?

    private:
        
        // Position label (e.g. 'anterior')
        // std::string
    };
    
    // Required since this a base class w/ pure virtual destructor 
    // (Meyers, Effective C++, 2nd Ed. pg. 63)
    inline Position::~Position() {} 
    
} // namespace datatypes




#endif	/* POSITION_H */

