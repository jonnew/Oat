//******************************************************************************
//* File:   Helpers.h
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

#ifndef OAT_SHMEMDFHELPERS_H
#define	OAT_SHMEMDFHELPERS_H

#include <algorithm>
#include <cassert>

#include "Source.h"

namespace oat {

template<typename T>
struct NamedSource {

    explicit NamedSource(const std::string &name,
                         std::unique_ptr<oat::Source<T>> &&source) :
      name(name),
      source(std::move(source))
    {
        // Nothing
    }

    const std::string name;
    std::unique_ptr<oat::Source<T>> source;
};

template<typename T>
using NamedSourceList = std::vector<NamedSource<T>>;

inline bool checkSamplePeriods(const std::vector<double> &periods,
                               double &min_rate) {

    assert(periods.size() > 0);
    min_rate = 1.0 / *std::max_element(std::begin(periods), std::end(periods));

    if (periods.size() > 1 &&
        !std::equal(periods.begin() + 1, periods.end(), periods.begin())) {
        return false;
    } else {
        return true;
    }
}

}       /* namespace oat */
#endif	/* OAT_SHMEMDFHELPERS_H */

