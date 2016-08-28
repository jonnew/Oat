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
#include <cmath>

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

/**
 * @brief Check if a set of sample periods is consistent.
 * @param periods_sec Sample periods in seconds.
 * @param min_rate The minimal sample rate in the set.
 * @param epsilon Equality tolerance.
 * @return True if maximum difference between periods is within epsilon. False
 * otherwise.
 */
inline 
bool checkSamplePeriods(const std::vector<double> &periods_sec,
                        double &min_rate,
                        double const epsilon=1e-6) {

    assert(periods_sec.size() > 0);

    double max_period = *std::max_element(std::begin(periods_sec), 
                                          std::end(periods_sec));
    min_rate = 1.0 / max_period;

    if (periods_sec.size() > 1) {
        for (auto &p : periods_sec)
            if (std::fabs(p - max_period) > epsilon)
                return false;
    }

    return true;
}

inline 
std::string inconsistentSampleRateWarning(double min_rate) {

    return 
        "WARNING: sample rates of sources are inconsistent.\n"
        " (1) This component forces synchronization at the lowest\n"
        "     source sample rate, which is " + std::to_string(min_rate) + "\n"
        " (2) You should probably use separate components to\n" 
        "     process these sources if you want independent rates.\n";
}

}       /* namespace oat */
#endif	/* OAT_SHMEMDFHELPERS_H */
