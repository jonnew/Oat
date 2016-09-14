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

#include "Sample.h"

namespace oat {

enum class PixelColor : int {
    mono8 = 0,
    color8,
    any
};

inline int cv_type(oat::PixelColor col) {
    switch (col) {
        case PixelColor::mono8 : return CV_8UC1;
        case PixelColor::color8 : return CV_8UC3;
        case PixelColor::any : // Fallthrough
        default : return -1;
    }
}

inline std::string to_string(oat::PixelColor col) {
    switch (col) {
        case PixelColor::mono8 : return "mono-8";
        case PixelColor::color8 : return "color-8";
        case PixelColor::any : return "any-color";
        default : return "";
    }
}

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

public:

    Frame() :
      cv::Mat()
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    explicit Frame(const double ts_sec) :
      cv::Mat()
    , sample_(ts_sec)
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

    Frame(cv::Mat m, const double ts_sec) :
      cv::Mat(m)
    , sample_(ts_sec)
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    Frame(cv::Mat m, const cv::Rect &roi) :
      cv::Mat(m, roi)
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    Frame(cv::Mat m, const cv::Rect &roi, const double ts_sec) :
      cv::Mat(m, roi)
    , sample_(ts_sec)
    , sample_ptr_(&sample_)
    {
        // Nothing
    }

    Frame(int r, int c, int t, void * data, void * samp_ptr) :
      cv::Mat(r, c, t, data)
    , sample_ptr_(static_cast<Sample *>(samp_ptr))
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

    // ROI
    Frame operator()(const cv::Rect &roi) const { return Frame(*this, roi); }

    // Expose sample information
    oat::Sample & sample() const { return *sample_ptr_; };

    // Provide copy of sample_
    oat::Sample sample_copy() const { return *sample_ptr_; };

private:

    // Internal Sample
    oat::Sample sample_;

    // sample_ptr_ can point to either outside data (shmem) or sample_
    oat::Sample * sample_ptr_;

    // Color profile of each pixel
    PixelColor color {PixelColor::any};
};

}      /* namespace oat */
#endif /* OAT_FRAME_H */
