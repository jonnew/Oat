//******************************************************************************
//* File:   DetectorFunc.cpp
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
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
//****************************************************************************

#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "../../lib/datatypes/Pose.h"

#include "DetectorFunc.h"

namespace oat {

void siftContours(cv::Mat &frame,
                  Pose &pose,
                  double &area,
                  double min_area,
                  double max_area)
{

    std::vector<std::vector <cv::Point> > contours;

    cv::findContours(frame, contours,
                     cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    double object_area = 0;

    for (auto &c : contours) {

        cv::Moments moment = cv::moments(static_cast<cv::Mat>(c));
        double countour_area = moment.m00;

        // Isolate the largest contour within the min/max range.
        if (countour_area >= min_area &&
            countour_area < max_area &&
            countour_area > object_area) {

            auto x = moment.m10 / countour_area;
            auto y = moment.m01 / countour_area;
            pose.found = true;
            auto p = std::array<double, 3>{{x, y, 0}};
            pose.set_position(p);

            object_area = countour_area;
        }
    }

    area = object_area;
}

} /* namespace oat */
