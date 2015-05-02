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

namespace datatypes {

    enum coordinate_system_type {
        PIXELS = 0, WORLD = 1
    };

    struct Position {
    
        Position(std::string position_label) : label(position_label) { }
      
        // Positions use one of two coordinate systems 
        // PIXELS - units are referenced to the sensory array of the digital camera. 
        //          origin in the upper left hand corner.
        // WORLD  - Defined by the homography matrix.
        int coord_system = PIXELS;

        // Positions must be able to convert themselves to world
        // coordinate system if they contain the appropriate
        // homography matracies
        //virtual Position convertToWorldCoordinates(void) = 0;
        // TODO: why can't this base class use the more general Position type instead
        // of Position2D etc?
        
        //Accessors
        std::string get_label(void) { return label; }

    private:
        
        // Position label (e.g. 'anterior')
        std::string label;
    };
} // namespace datatypes


#endif	/* POSITION_H */

