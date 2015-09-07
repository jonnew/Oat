//******************************************************************************
//* File:   IOUtility.h
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

#ifndef IOUTILITY_H
#define IOUTILITY_H

#include <iostream>

namespace oat {

    /**
     * Used to clear the input stream after bad input is detected.  Ignores
     * stream content up to a newline.
     */
    void ignoreLine(std::istream& in) {

        in.clear();
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    /**
     * Used to clear the input stream after bad input is detected.  Ignores
     * all stream content.
     */
    void ignoreAll(std::istream& in) {

        in.clear();
        in.ignore(std::numeric_limits<std::streamsize>::max());
    }
}

#endif // IOUTILITY_H
