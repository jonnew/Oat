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

        Position() { };

        // TODO: should label version be explicit?
        Position(const std::string& label) :
            label_{*label.data()} {
        }

        virtual ~Position() { };

        // Positions use one of two coordinate systems
        DistanceUnit unit_of_length {DistanceUnit::PIXELS};

        // sample_ only settable via incrementation
        uint64_t incrementSampleCount() { return ++sample_; }
        
        // Accessors
        uint64_t sample() const { return sample_; }

    protected:

        char label_[100] {'N', 'A'}; //!< Position label (e.g. "anterior")

        uint64_t sample_ {0};
    };

}      /* namespace oat */
#endif /* OAT_POSITION_H */

