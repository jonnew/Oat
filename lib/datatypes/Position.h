//******************************************************************************
//* File:   Position.h 
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
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

#ifndef POSITION_H
#define	POSITION_H

namespace oat {

    enum coordinate_system_type {
        PIXELS = 0, WORLD = 1
    };

    class Position {

    public:
        
        // TODO: How can label be incorporated as an argument to the Position constructor
        //       without screwing up all the class templates that use Position as type parameter?
        Position() {
            std::strncpy(label_, "NA", sizeof(label_));
        } 
        
        virtual ~Position() = 0;
      
        // Positions use one of two coordinate systems 
        // PIXELS - units are referenced to the sensory array of the digital camera. 
        //          origin in the upper left hand corner.
        // WORLD  - Defined by the homography matrix.
        int coord_system {PIXELS};
        
        // Accessors
        inline void set_label(const std::string& value) { 
            std::strncpy(label_, value.c_str(), sizeof(label_));
        }
        
        inline void set_label(const uint32_t value) { 
            sample = value;
        }
        
    protected:
        
        //Position label (e.g. "anterior")
        char label_[100];
        
        // TODO: sample number should travel with the Position
        uint32_t sample;
    };
    
    // Required since this a base class w/ pure virtual destructor 
    // (Meyers, Effective C++, 2nd Ed. pg. 63)
    inline Position::~Position() {} 
    
} // namespace datatypes

#endif	/* POSITION_H */

