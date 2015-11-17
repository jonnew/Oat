//******************************************************************************
//* File:   PathChanger.h
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu)
//
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

#ifndef OAT_PATHCHANGER_H
#define OAT_PATHCHANGER_H

#include "CalibratorVisitor.h"

// Forward declarations
class Calibrator;
class CameraCalibrator;
class HomographyGenerator;

namespace oat {

/**
 *
 */
class PathChanger : public CalibratorVisitor {

public:

    /**
     * Change the camera calibration file save path.
     */
    void visit(CameraCalibrator* cc) override;

    /**
     * Change the homography calibration file save path.
     */
    void visit(HomographyGenerator* hg) override;

private:

    void setNewPath(Calibrator* cal);
};

}      /* namespace oat */
#endif /* OAT_PATHCHANGER_H */
