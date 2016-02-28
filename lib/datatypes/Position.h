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

#ifndef OAT_POSITION_H
#define	OAT_POSITION_H


#include "Sample.h"

namespace oat {

/**
 * Unit of length used to specify position.
 */
enum class DistanceUnit
{
    PIXELS = 0,   //!< Position measured in pixels. Origin is upper left.
    WORLD = 1     //!< Position measured in units specified via homography
};

class Position {

public:

    Position(const std::string &label) 
    {
        strncpy(label_, label.c_str(), sizeof(label_));
        label_[sizeof(label_) - 1] = 0;
    }

    virtual ~Position() { };

    Position & operator = (const Position &p) {

        // Check for self assignment
        if (this == &p)
            return *this;

        // Copy all except label_
        unit_of_length_ = p.unit_of_length_;
        sample_ = p.sample_;
        return *this;
    }

    // Expose sample information
    oat::Sample & sample() { return sample_; };

    // Accessors
    char * label() {return label_; }
    DistanceUnit unit_of_length(void) const { return unit_of_length_; }
    
protected:
    
    char label_[100]; //!< Position label (e.g. "anterior")
    DistanceUnit unit_of_length_ {DistanceUnit::PIXELS};
    
    oat::Sample sample_;
};

}      /* namespace oat */
#endif /* OAT_POSITION_H */

