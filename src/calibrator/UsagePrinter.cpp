//******************************************************************************
//* File:   UsagePrinter.cpp
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

#include "CameraCalibrator.h"
#include "HomographyGenerator.h"
#include "UsagePrinter.h"

#include <iostream>

namespace oat {

void UsagePrinter::visit(CameraCalibrator*, std::ostream& out) {

    out << "COMMANDS\n"
        << "(To use, make sure the display window is in focus.)\n\n"
        << "CMD    FUNCTION\n"
        << "  d    Toggle to and from chessboard detection mode. When in\n"
        << "       detection mode, incoming frames are searched for the presence\n"
        << "       of a chessboard pattern of specified height and width. When\n"
        << "       detected, a multicolor grid will appear at each corner of the\n"
        << "       board and corner locations are saved to form a data set for\n"
        << "       calculating the camera matrix and distortion coefficients.\n"
        << "       corner locations are saved at a maximum rate of 1 Hz.\n"
        << "  f    Specify the calibration file save path to which the\n"
        << "       camera matrix and distortion coefficients will be saved.\n"
        << "  g    Generate camera matrix and distortion coefficients using the\n"
        << "       current set of chessboard corner locations. If successful,\n"
        << "       the camera matrix, distortion coefficients, and RMS\n"
        << "       reconstruction error will be printed.\n"
        << "  h    Print this information.\n"
        //<< "  m    Specify the camera model used to generate the calibration\n"
        //<< "       from this list:\n"
        //<< "        - pinhole (default): camera modeled as a lensless aperture.\n"
        //<< "        - fisheye: ultra wide-angle lens with strong visual\n"
        //<< "          intended to create a wide panoramic or hemispherical image.\n"
        << "  p    Print the current calibration results.\n"
        << "  u    Apply results of camera calibration to undistort the frame\n"
        << "       stream shown in the display window.\n"
        << "  s    Save the camera calibration to the specified calibration file\n"
        << "       If the file exists, this will create or modify the\n"
        << "       'calibration-<subfield>' entries. If not, the file will be\n"
        << "       created. Other fields within an existing file not be affected.\n\n";
}

void UsagePrinter::visit(HomographyGenerator*, std::ostream& out) {

    out << "COMMANDS\n"
        << "(To use, make sure the display window is in focus.)\n\n"
        << "CMD    FUNCTION\n"
        << "  a    Add world-coordinates for the currently selected pixel and\n"
        << "       append to the pixel-to-world coordinate data set. Make sure\n"
        << "       you have clicked a point on the display window to select a\n"
        << "       pixel prior to using this command.\n"
        << "  d    Remove an entry from the pixel-to-world coordinate data set\n"
        << "       using its index. The 'p' command shows the index of each data\n"
        << "       entry.\n"
        << "  f    Specify the calibration file save path to which the\n"
        << "       homography will be saved.\n"
        << "  g    Generate a homography using the current pixel-to-world data\n"
        << "       set. If successful, both the pixel and world coordinate will\n"
        << "       be shown for selected pixels on the display window and the\n"
        << "       homography matrix will be printed.\n"
        << "  h    Print this information.\n"
        << "  m    Specify the homography estimation procedure from this list:\n"
        << "        - robust (default): RANSAC-based robust estimation method\n"
        << "          (automatic outlier rejection).\n"
        << "        - regular: Best-fit using all data points.\n"
        << "        - exact: Compute the homography that exactly fits four points.\n"
        << "          Useful when frames contain precisely known fiducial marks.\n"
        << "  p    Print the current pixel-to-world coordinate data set.\n"
        << "  s    Save the homography to the specified calibration file.\n"
        << "       If the file exists, this will create or modify the 'homography'\n"
        << "       entry. If not, the file will be created. Other fields within\n"
        << "       an existing file not be affected.\n\n";
}

} /* namespace oat */
