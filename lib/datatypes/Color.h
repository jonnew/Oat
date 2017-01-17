//******************************************************************************
//* File:   Color.h
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

#ifndef OAT_COLOR_H
#define	OAT_COLOR_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace oat {

enum PixelColor {
    PIX_BINARY = 0,
    PIX_GREY,
    PIX_BGR, // Default
    PIX_HSV,
    PIX_ANY
};

// Used conversion structures
static const int color_2_cvtype[4]{CV_8UC1, CV_8UC1, CV_8UC3, CV_8UC3};
static const int color_2_bytes[4]{1, 1, 3, 3};

static const int color_2_imread_code[4]{
    -2, cv::IMREAD_GRAYSCALE, cv::IMREAD_COLOR, -2};

// Arguments are from/to PixelColors
// -1 = No conversion needed
// -2 = Conversion not possible
static const int color_conv_table[4][4]{
    {-1, -1, cv::COLOR_GRAY2BGR, -2}, // From BINARY
    {-1, -1, cv::COLOR_GRAY2BGR, -2}, // From GREY
    {cv::COLOR_BGR2GRAY, cv::COLOR_BGR2GRAY, -1, cv::COLOR_BGR2HSV}, // From BGR
    {-2, -2, cv::COLOR_HSV2BGR, -1}, // From HSV
};

inline std::string color_str(const oat::PixelColor col)
{
    switch (col) {
        case PIX_BINARY : return "BINARY";
        case PIX_GREY : return "GREY";
        case PIX_BGR : return "BGR";
        case PIX_HSV : return "HSV";
        default : throw std::runtime_error("Invalid color.");
    }
}

inline oat::PixelColor str_color(const std::string &s)
{
    if (s == "BINARY")
        return PIX_BINARY;
    else if (s == "GREY")
        return PIX_GREY;
    else if (s == "BGR")
        return PIX_BGR;
    else if (s == "HSV")
        return PIX_HSV;
    else
        throw std::runtime_error("Invalid color.");
}

inline int cv_type(oat::PixelColor col)
{
    return color_2_cvtype[col];
}

inline int color_bytes(oat::PixelColor col)
{
    return color_2_bytes[col];
}

inline int color_conv_code(oat::PixelColor from, oat::PixelColor to)
{
    return color_conv_table[from][to];
}

inline int imread_code(oat::PixelColor col)
{
    auto code = color_2_imread_code[col];
    if (code == -2) {
        throw std::runtime_error(
            "Images cannot be loaded using the requested color format.");
    }
    return code;
}

}      /* namespace oat */
#endif /* OAT_COLOR_H */
