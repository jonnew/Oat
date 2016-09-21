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

    using USec = Sample::Microseconds;

public:
    explicit Position(const std::string &label)
    {
        strncpy(label_, label.c_str(), sizeof(label_));
        label_[sizeof(label_) - 1] = '\0';
    }

    virtual ~Position() { };

    Position &operator=(const Position &p)
    {

        // Check for self assignment
        if (this == &p)
            return *this;

        // Copy all except label_
        unit_of_length_ = p.unit_of_length_;
        sample_ = p.sample_;
        position_valid = p.position_valid;
        velocity_valid = p.velocity_valid;
        heading_valid = p.heading_valid;
        region_valid = p.region_valid;
        strncpy(region, p.region, sizeof(region));
        region[sizeof(region) - 1] = '\0';

        return *this;
    }

    // Expose sample information for potential modification
    // TODO: This is very ugly: why even make it protected??
    //oat::Sample & sample() { return sample_; };

    // Accessors
    char *label() { return label_; }
    DistanceUnit unit_of_length(void) const { return unit_of_length_; }

    // Categorical position
    bool region_valid {false};
    char region[100] {0}; //!< Categorical position label (e.g. "North West")

    // Validity booleans
    bool position_valid {false};
    bool velocity_valid {false};
    bool heading_valid {false};

    // Set sample rate
    void set_sample(const Sample &val) { sample_ = val; }
    void set_rate_hz(const double rate_hz) { sample_.set_rate_hz(rate_hz); }
    double sample_period_sec() const { return sample_.period_sec().count(); }
    uint64_t sample_count(void) const { return sample_.count(); }
    void incrementSampleCount() { sample_.incrementCount(); }
    void incrementSampleCount(USec us) { sample_.incrementCount(us); }

protected:
    char label_[100] {0}; //!< Position label (e.g. "anterior")
    DistanceUnit unit_of_length_ {DistanceUnit::PIXELS};

    oat::Sample sample_;
};

}      /* namespace oat */
#endif /* OAT_POSITION_H */
