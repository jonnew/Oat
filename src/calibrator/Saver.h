//******************************************************************************
//* File:   Saver.h
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

#ifndef SAVER_H
#define SAVER_H

#include "CalibratorVisitor.h"

// Forward declarations
class CameraCalibrator;
class HomographyGenerator;

/**
 *
 */
class Saver : public CalibratorVisitor {

public:

    void visit(CameraCalibrator* cc) override;
    void visit(HomographyGenerator* hg) override;

private:

    cpptoml::datetime generateDateTime();
    cpptoml::table generateCalibrationTable(const std::string& file , const std::string& key);
    void saveCalibrationTable(const cpptoml::table& table, const std::string& file);

};

#endif // SAVER_H
