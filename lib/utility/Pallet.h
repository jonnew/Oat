//******************************************************************************
//* File:   Colors .h
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
//* but WITHOUcv::Scalar ANY WARRANTY; without even the implied warranty of
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//* GNU General Public License for more details.
//* You should have received a copy of the GNU General Public License
//* along with this source code.  If not, see <http://www.gnu.org/licenses/>.
//******************************************************************************

#ifndef OAT_PALETTE_H
#define	OAT_PALETTE_H

#include <opencv2/core.hpp>

namespace oat {

enum class Roygbiv {
    red = 0,
    orange,
    yellow,
    lime,
    green,
    aqua,
    blue,
    lightblue,
    indigo,
    violet,
    pink,
    count
};

enum class MixedPallet {
    red = 0,
    blue,
    green,
    orange,
    lightblue,
    pink,
    aqua,
    violet,
    yellow,
    indigo,
    lime,
    count
};

template<typename P = Roygbiv>
struct RGB {

    RGB() = default;

    cv::Scalar next()
    {
        const auto c = color(static_cast<P>(idx_));
        if (++idx_ == static_cast<size_t>(P::count))
            idx_ = 0;

        return c;
    }

    static cv::Scalar white()
    {
        return CV_RGB(255, 255, 255);
    }

    static cv::Scalar black()
    {
        return CV_RGB(0, 0, 0);
    }

    static cv::Scalar color(P color)
    {
        switch (color) {
            case P::red:
                return CV_RGB(255, 51, 51);
            case P::orange:
                return CV_RGB(255, 153, 51);
            case P::yellow:
                return CV_RGB(255, 255, 51);
            case P::lime:
                return CV_RGB(153, 255, 51);
            case P::green:
                return CV_RGB(51, 255, 51);
            case P::aqua:
                return CV_RGB(51, 255, 153);
            case P::blue:
                return CV_RGB(51, 51, 255);
            case P::lightblue:
                return CV_RGB(51, 153, 255);
            case P::indigo:
                return CV_RGB(51, 255, 255);
            case P::violet:
                return CV_RGB(153, 51, 255);
            case P::pink:
                return CV_RGB(255, 51, 153);
            case P::count:
                throw std::runtime_error("Bad color choice.");
        }
    }

private:
    size_t idx_{0};
};

}      /* namespace oat */
#endif /* OAT_PALETTE_H */
