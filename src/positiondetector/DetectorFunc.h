//******************************************************************************
//* File:   DetectorFunc.h
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

#ifndef OAT_DETECTORFUNC
#define	OAT_DETECTORFUNC

#include <opencv2/highgui.hpp>

// Forward decl.
namespace cv { class Mat; }

namespace oat {

// Constants
static constexpr double PI {3.14159265358979323846};

// Forward decl.
class Pose;

/**
 * Given a binary frame, find all contours and return a position corresponding
 * to the centroid of the largest one.
 * @param frame_in Frame to look for positions in.
 * @note This function will modify the frame.
 * @param pose Pose output
 * @param min_area Minimum contour area to be considered candidate for position
 * @param max_area Maximum contour area to be considered candidate for position
 * @return Position corresponding the centroid of the largest contour in the frame.
 */
void siftContours(cv::Mat &frame,
                  Pose &pose,
                  double &object_area,
                  double min_area,
                  double max_area);

}       /* namespace oat */
#endif	/* OAT_DETECTORFUNC */
