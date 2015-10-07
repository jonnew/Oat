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

    /**
     * Unit of length used to specify position.
     */
    enum length_unit
    {
        PIXELS = 0,   //!< Position measured in pixels. Origin is upper left.
        WORLD = 1     //!< Position measured in units specified via homography
    };

    class Position {

    public:

        Position() { };

        Position(const std::string& label) :
            label_{*label.data()} {}

        virtual ~Position();

        // Positions use one of two coordinate systems
        int coord_system {PIXELS};

        inline void set_sample(const uint32_t value) {
            sample = value;
        }

    protected:

        char label_[100] {'N', 'A'}; //!< Position label (e.g. "anterior")

        // TODO: sample number should be managed globally for a data processing
        // chain
        uint32_t sample {0};
    };

} // namespace datatypes

#endif	/* POSITION_H */

