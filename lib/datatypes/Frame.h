//******************************************************************************
//* File:   Frame.h
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

#ifndef OAT_FRAME_H
#define	OAT_FRAME_H

#include <opencv2/core/mat.hpp>

namespace oat {

// TODO: cv::Mat's destructor is not virtual! Might be best just use
// cv::Mat as a public member
class Frame : public cv::Mat {

public:

    Frame() { }
    Frame(cv::Mat m) : cv::Mat(m) { }

    // Clone override
    Frame clone() const {
        cv::Mat m;
        copyTo(m);

        Frame f(m); 
        return Frame(f);
    }

    // sample_ only settable via incrementation
    void incrementSampleCount() { ++sample_; }

    // Accessors
    double sample_period_sec(void) const { return sample_period_sec_; }
    void set_sample_period_sec(double value) { sample_period_sec_ = value; }
    uint64_t sample(void) const { return sample_; }

private:
    uint64_t sample_ {0};
    double sample_period_sec_ {0.0};
};

}      /* namespace oat */
#endif /* OAT_FRAME_H */

