//******************************************************************************
//* File:   Token.h
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

#ifndef OAT_TOKEN_H
#define	OAT_TOKEN_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <ratio>

namespace oat {

/**
 * Token data object holding sample timing information and nothing else.
 */
class Token {

public:
    using Seconds = std::chrono::duration<double, std::ratio<1>>;
    using Microseconds = std::chrono::microseconds;
    using IEEE1394Tick = std::chrono::duration<float, std::ratio<1,8000>>;

    explicit Token()
    {
        // Nothing
    }

    /**
     * @brief Generic, timed data sample.
     * @param period_sec The period of the sample clock in seconds.
     */
    explicit Token(const double period_sec)
    : period_(period_sec)
    {
        assert(period_.count() > 0.0 && "Sample period is 0 or negative.");
    }

    /**
     * @brief Generic, timed data sample.
     * @param period_sec The period of the sample clock in seconds.
     */
    explicit Token(const Seconds period_sec)
    : period_(period_sec)
    {
        assert(period_.count() > 0.0 && "Sample period is 0 or negative.");
    }

    /**
     * @brief Increment sample count with time override. 
     * @note Only pure SINKs should increment the count, set the sample rates,
     * periods, etc.
     * @param step Number of ticks to increment count
     * @param time Manual override of current sample time, e.g. from an
     * external clock reading.
     * @return Current sample count
     */
    template <typename DurationT = Token::Seconds>
    uint64_t incrementCount(uint64_t step = 1, DurationT time = Token::Seconds(0))
    {
        if (time == Token::Seconds(0))
            time_ += period_;
        else
            time_ = std::chrono::duration_cast<Seconds>(time);

        tick_ += step;
        return tick_;
    }

    template <typename DurationT = Token::Seconds>
    void setTime(uint64_t tick, const DurationT time)
    {
        tick_ = tick;
        time_ = std::chrono::duration_cast<Seconds>(time);
    }

    ///**
    // * @brief Set the sample rate.
    // * @param value Token rate in Hz.
    // */
    //void set_rate_hz(const double value)
    //{
    //    assert(value > 0.0);
    //    period_ = Seconds(1.0 / value);
    //}


    uint64_t tick() const { return tick_; }

    template <typename DurationT>
    DurationT time() const
    {
        return std::chrono::duration_cast<DurationT>(time_);
    }

    template <typename DurationT = Token::Seconds>
    DurationT period() const
    {
        return std::chrono::duration_cast<DurationT>(period_);
    }

    template <typename DurationT = Token::Seconds>
    void set_period(const DurationT value) { period_ = value; }

    //double rate_hz() const { return 1.0 / period_.count(); }

    void resample(const unsigned int l,
                  const unsigned int m,
                  const unsigned int o = 0)
    {
        period_ /= l;
        period_ *= m;
        tick_ *= l;
        tick_ /= m;
        tick_ += o;
    }

private:
    uint64_t tick_{0};
    Seconds time_{0};
    Seconds period_{0.0};
};

}      /* namespace oat */
#endif /* OAT_TOKEN_H */
