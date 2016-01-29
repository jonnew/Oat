//******************************************************************************
//* File:   Sample.h
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

#ifndef OAT_SAMPLE_H
#define	OAT_SAMPLE_H

#include <algorithm>
#include <chrono>
#include <ratio>

#include <opencv2/core/mat.hpp>

namespace oat {

/**
 * Class specifiying general sample timing information.
 */
class Sample {

public:

    //using Clock = std::chrono::system_clock;
    using Microseconds = std::chrono::microseconds; //duration<int64_t, std::micro>;
    //using Time = std::chrono::time_point<Clock, Milliseconds>;
    using IEEE1394Tick = std::chrono::duration<float, std::ratio<1,8000>>;

    Sample() 
    {
        // Nothing
    }

    /**
     * An incrementable sample keeper.
     *
     * @param period_sec The period of the sample clock in seconds.
     */
    Sample(double period_sec) : 
      period_sec_(period_sec)
    , rate_hz_( 1.0 / period_sec)
    {
        // Nothing
    }

    // Only pure SINKs should increment the count, set the sample rates, periods, etc
    uint64_t incrementCount() { 
        return ++count_; 
    }

    uint64_t incrementCount(const Microseconds usec) { 
        microseconds_ = usec; 
        return ++count_; 
    }

    void set_rate_hz(const double value) { 
        rate_hz_ = value;
        period_sec_ = 1.0 / value;
    }

    void set_period_sec(const double value) { 
        period_sec_ = value;
        rate_hz_ = 1.0 / value;
    }

    uint64_t count() const { return count_; }
    Microseconds microseconds() const { return microseconds_; }
    double period_sec() const { return period_sec_; }
    double rate_hz() const { return 1.0 / period_sec_; }
    
private:

    uint64_t count_ {0};
    Microseconds microseconds_;
    double period_sec_ {-1.0};
    double rate_hz_ {-1.0};

};

}      /* namespace oat */
#endif /* OAT_SAMPLE_H */


