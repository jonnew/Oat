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

#include <algorithm>

#include <opencv2/core/mat.hpp>

namespace oat {

/**
 * Wrapper class for cv::Mat that contains sample number information.
 *
 * NOTE 1: cv::Mat does not declare a virtual destructor, so you must not
 * delete Frames through a pointer to the base cv::Mat.
 *
 * NOTE 2: Only the cv::Mat operators and constructors that are specifically needed by the
 * oat codebase are overriden in this class. Others remain callable, and might
 * not make any sense or result in object slicing.
 */
class Frame : public cv::Mat {

public:

    Frame() :
      cv::Mat()
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    Frame(cv::Mat m, const cv::Rect &roi) :
      cv::Mat(m,roi)
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    Frame(cv::Mat m) :
      cv::Mat(m)
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    Frame(int r, int c, int t, void * data, void * samp_ptr) :
      cv::Mat(r, c, t, data)
    , sample_ptr_((Sample *)(samp_ptr))
    {
        // Nothing
    }

    Frame clone() const {
        Frame f(cv::Mat::clone());
        *(f.sample_ptr_) = *sample_ptr_;
        return f;
    }

    void copyTo(Frame &f) const {
        cv::Mat::copyTo(f);
        *(f.sample_ptr_) = *sample_ptr_;
    }

    Frame operator()( const cv::Rect &roi ) const {
        return Frame(*this, roi);
    }

    // sample_count_ only settable via incrementation
    void incrementSampleCount() { ++(sample_ptr_->sample_count_); }

    // Accessors
    double sample_period_sec(void) const { return sample_ptr_->sample_period_sec_; }
    void set_sample_period_sec(double value) { sample_ptr_->sample_period_sec_ = value; }
    uint64_t sample_count(void) const { return sample_ptr_->sample_count_; }

    struct Sample {
        uint64_t sample_count_    {0};
        double sample_period_sec_ {0.0};
    };

private:

    // sample_ptr_ can point either to outside data (shmem) or internal
    // sample_ object.
    Sample * sample_ptr_;
    Sample sample_;
};

}      /* namespace oat */
#endif /* OAT_FRAME_H */

