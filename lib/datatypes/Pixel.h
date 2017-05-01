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

#ifndef OAT_PIXEL_H
#define	OAT_PIXEL_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace oat {

struct Pixel {

    using ValueT = unsigned char;

    enum class Color {
        binary = 0,
        mono,
        bgr, // default
        hsv
    };

    // TODO:C++14 allows this to be constexpr return value if runtime_error is
    // replaced with static_assert
    //
    /** 
     * @brief Get a string version of a pixel color.
     */
    static inline std::string string(const Color col)
    {
        switch (col) {
            case Color::binary:
                return "binary";
            case Color::mono:
                return "mono";
            case Color::bgr:
                return "bgr";
            case Color::hsv:
                return "hsv";
            default:
                throw std::runtime_error("Invalid color.");
        }
    }


    /** 
     * @brief Is the pixel color non-monochromatic?
     * @param col Color to test
     * @return True if non-monochromatic. False otherwise.
     */
    static inline bool multichomatic(const Color col)
    {
        switch (col) {
            case Color::binary:
            case Color::mono:
                return false;
            case Color::bgr:
            case Color::hsv:
                return true;
            default:
                throw std::runtime_error("Invalid color.");
        }
    }

    /** 
     * @brief Get byte-depth of a pixel color
     */
    static inline constexpr size_t depth(Color p)
    {
        return to_depth[static_cast<size_t>(p)];
    }

    // TODO:C++14 allows this to be constexpr return value if runtime_error is
    // replaced with static_assert
    /** 
     * @brief Convert a string to a pixel color.
     */
    static inline Color color(const std::string &s)
    {
        if (s == "binary")
            return Color::binary;
        else if (s == "mono")
            return Color::mono;
        else if (s == "bgr")
            return Color::bgr;
        else if (s == "hsv")
            return Color::hsv;
        else
            throw std::runtime_error("Invalid color.");
    }

    static inline constexpr int cvType(Color col)
    {
        return to_cvtype[static_cast<int>(col)];
    }

    static inline Color color(int cvtype)
    {
        switch (cvtype) {
            case CV_8UC1:
                return Color::mono;
            case CV_8UC3:
                return Color::bgr;
            default:
                throw std::runtime_error("Invalid cv type.");
        }
    }

    static inline constexpr size_t bytes(Color col)
    {
        return to_bytes[static_cast<int>(col)];
    }

    /** 
     * @brief Get the opencv conversion code to transform between pixel colors.
     * @param from
     * @param to
     * @return OpenCV conversion code.
     */
    static inline constexpr int cvConvCode(Color from, Color to)
    {
        return color_conv_table[static_cast<int>(from)][static_cast<int>(to)];
    }

    // TODO: C++14 allows this to be constexpr return value
    static inline int cvImreadCode(Color col)
    {
        const auto code = to_imread_code[static_cast<int>(col)];
        if (code == -2) {
            throw std::runtime_error(
                "Images cannot be loaded using the requested color format.");
        }
        return code;
    }

private:
    static constexpr size_t to_depth[4]{1, 1, 3, 3};
    static constexpr int to_imread_code[4]{ -2, cv::IMREAD_GRAYSCALE, cv::IMREAD_COLOR, -2};
    static constexpr size_t to_bytes[4]{1, 1, 3, 3};
    static constexpr int to_cvtype[4]{CV_8UC1, CV_8UC1, CV_8UC3, CV_8UC3};
    //static constexpr int from_cvtype[2]{Color::mono, Color::bgr};
    // Arguments are from/to Colors
    // -1 = No conversion needed
    // -2 = Conversion not possible
    static constexpr int color_conv_table[4][4]{
        {-1, -1, cv::COLOR_GRAY2BGR, -2}, // From BINARY
        {-1, -1, cv::COLOR_GRAY2BGR, -2}, // From GREY
        {cv::COLOR_BGR2GRAY, cv::COLOR_BGR2GRAY, -1, cv::COLOR_BGR2HSV}, // From BGR
        {-2, -2, cv::COLOR_HSV2BGR, -1}, // From HSV
    };
};

}      /* namespace oat */
#endif /* OAT_PIXEL_H */
