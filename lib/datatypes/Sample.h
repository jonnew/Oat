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
#include <cassert>
#include <chrono>
#include <ratio>

#include <opencv2/core/mat.hpp>

namespace oat {

/**
 * Class specifying general sample timing information.
 */
class Sample {

public:

    using Seconds = std::chrono::duration<double, std::ratio<1>>;
    using Microseconds = std::chrono::microseconds;
    using IEEE1394Tick = std::chrono::duration<float, std::ratio<1,8000>>;

    explicit Sample()
    {
        // Nothing
    }

    /**
     * @brief An increment-able sample keeper.
     * @param period_sec The period of the sample clock in seconds.
     */
    explicit Sample(const double period_sec)
    : period_sec_(period_sec)
    {
        assert(period_sec_.count() > 0.0);
        rate_hz_ = 1.0 / period_sec;
    }

    /**
     * @brief An increment-able sample keeper.
     * @param period_sec The period of the sample clock in seconds.
     */
    explicit Sample(const Seconds period_sec)
    : period_sec_(period_sec)
    {
        assert(period_sec_.count() > 0.0);
        rate_hz_ = 1.0 / period_sec.count();
    }

    /**
     * @brief Increment sample count. Only pure SINKs should increment the
     * count, set the sample rates, periods, etc.
     * @return Current sample count
     */
    uint64_t incrementCount()
    {
        microseconds_ += period_microseconds_;
        return ++count_;
    }

    /**
     * @brief Increment sample count with microseconds override. Only pure SINKs
     * should increment the count, set the sample rates, periods, etc.
     * @param usec Manual override of current sample time in microseconds, e.g.
     * from an external clock reading.
     * @return Current sample count
     */
    uint64_t incrementCount(const Microseconds usec)
    {
        microseconds_ = usec;
        return ++count_;
    }

    /**
     * @brief Set the sample rate.
     * @param value Sample rate in Hz.
     */
    void set_rate_hz(const double value)
    {
        assert(value > 0.0);
        rate_hz_ = value;
        period_sec_ = Seconds(1.0 / value);
        period_microseconds_ =
            std::chrono::duration_cast<Microseconds>(period_sec_);
    }

    uint64_t count() const { return count_; }
    Microseconds microseconds() const { return microseconds_; }
    Seconds seconds() const { return std::chrono::duration_cast<Seconds>(microseconds_); }
    Seconds period_sec() const { return period_sec_; }
    Microseconds period_microseconds() const { return period_microseconds_; }
    double rate_hz() const { return rate_hz_; }

    void resample(const double resample_ratio) {
        set_rate_hz(rate_hz_ * resample_ratio);
        count_ *= resample_ratio;
    }

private:

    uint64_t count_ {0};
    Microseconds microseconds_ {0};
    Seconds period_sec_ {0.0};
    Microseconds period_microseconds_ {0};
    double rate_hz_ {0.0};
};

}      /* namespace oat */
#endif /* OAT_SAMPLE_H */
