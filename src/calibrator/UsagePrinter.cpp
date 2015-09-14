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

#include <iostream>

#include "CameraCalibrator.h"
#include "HomographyGenerator.h"
#include "UsagePrinter.h"

void UsagePrinter::visit(CameraCalibrator* camera_calibrator, std::ostream& out) {

}

void UsagePrinter::visit(HomographyGenerator* hg, std::ostream& out) {

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
        << "       set.\n"
        << "       If successful, both the pixel and world coordinate will be\n"
        << "       shown for selected pixels on the display window and the\n"
        << "       homography matrix will be printed.\n"
        << "  h    Print this information.\n"
        << "  m    Specify the homography estimation procedure from this list:\n"
        << "        - robust (default): RANSAC-based robust estimation method\n"
        << "          (automatic outlier rejection).\n"
        << "        - regular: Best-fit using all data points.\n"
        << "        - exact: Compute the homography that exactly fits four points.\n"
        << "          Useful when frames contain precisely know fiducial marks.\n"
        << "  p    Print the current pixel-to-world coordinate data set.\n"
        << "  s    Save the homography to the specified calibration file.\n"
        << "       If the file exists, this will modify the 'homography' entry\n"
        << "       in the file or create the 'homographgy' entry if it does not\n"
        << "       exist. Other fields within an existing file not be affected.\n"
        << "       If the file does not exist, it will be created.\n\n";
}

