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
#include <opencv2/imgproc.hpp>

#include "Color.h"
#include "Sample.h"

namespace oat {

/**
 * Wrapper class for cv::Mat that contains sample number information.
 *
 * NOTE 1: cv::Mat does not declare a virtual destructor, so you must not
 * delete oat::Frames through a pointer to cv::Mat.
 *
 * NOTE 2: Only the cv::Mat operators and constructors that are specifically needed by the
 * oat codebase are overriden in this class. Others remain callable, and might
 * not make any sense or result in object slicing.
 */
class Frame : public cv::Mat {

using USec = Sample::Microseconds;

public:

    Frame()
    : cv::Mat()
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    explicit Frame(const double ts_sec)
    : cv::Mat()
    , sample_(ts_sec)
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    explicit Frame(cv::Mat m)
    : cv::Mat(m)
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    Frame(cv::Mat m, const double ts_sec)
    : cv::Mat(m)
    , sample_(ts_sec)
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    Frame(cv::Mat m, const cv::Rect &roi)
    : cv::Mat(m, roi)
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    Frame(cv::Mat m, const cv::Rect &roi, const double ts_sec)
    : cv::Mat(m, roi)
    , sample_(ts_sec)
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    Frame(const int r,
          const int c,
          const int t,
          const oat::PixelColor col,
          void *data,
          void *samp_ptr)
    : cv::Mat(r, c, t, data)
    , sample_ptr_(static_cast<Sample *>(samp_ptr))
    , color_(col)
    {
        // Nothing
    }

    Frame clone() const
    {
        Frame f(cv::Mat::clone());
        *(f.sample_ptr_) = *sample_ptr_;
        f.color_ = color_;
        return f;
    }

    void copyTo(Frame &f) const
    {
        cv::Mat::copyTo(f);
        *(f.sample_ptr_) = *sample_ptr_;
        f.color_ = color_;
    }

    void copyTo(Frame &f, const cv::Mat &mask) const
    {
        cv::Mat::copyTo(f, mask);
        *(f.sample_ptr_) = *sample_ptr_;
        f.color_ = color_;
    }

    // ROI
    Frame operator()(const cv::Rect &roi) const { return Frame(*this, roi); }

    // Set sample rate
    void set_rate_hz(const double rate_hz) { sample_ptr_->set_rate_hz(rate_hz); }
    double sample_period_sec() const { return sample_ptr_->period_sec().count(); }
    uint64_t sample_count(void) const { return sample_ptr_->count(); }
    void incrementSampleCount() { sample_ptr_->incrementCount(); }
    void incrementSampleCount(USec us) { sample_ptr_->incrementCount(us); }
    void resample(const double resample_ratio) { sample_.resample(resample_ratio); }

    // Provide copy of sample_
    oat::Sample sample() const { return *sample_ptr_; };

    // Color accessors
    PixelColor color(void) const { return color_; }
    void set_color(const PixelColor val) { color_ = val; }

private:
    // Internal Sample
    oat::Sample sample_;

    // sample_ptr_ can point to either outside data (shmem) or sample_
    oat::Sample * sample_ptr_;

    // Color profile of each pixel
    oat::PixelColor color_ {oat::PIX_BGR};
};

/**
 * @brief Frame color conversion.
 * @param from Source frame
 * @param to Result frame
 * @param color Desired color of result frame.
 */
inline void convertColor(const oat::Frame &from, oat::Frame &to, oat::PixelColor color)
{
    auto code = color_conv_code(from.color(), color);
    if (code == -2) {
        throw std::runtime_error("Requested color conversion is not possible.");
    } else if (code == -1) {
        return;
    } else {
        cv::cvtColor(from, to, code);
        to.set_color(color);
    }
}

}      /* namespace oat */
#endif /* OAT_FRAME_H */
