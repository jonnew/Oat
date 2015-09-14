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

    Saver(const std::string& entry_key, const std::string& calibration_file);

    /**
     * Save camera calibration parameters to file.
     */
    void visit(CameraCalibrator* cc) override;

    /**
     * Save camera homography to file.
     */
    void visit(HomographyGenerator* hg) override;

private:

    const std::string entry_key_;
    const std::string calibration_file_;

    /**
     * Generate a cpptoml datetime to update the last-modified entry of the
     * calibration file.
     */
    cpptoml::datetime generateDateTime();

    /**
     * Either retrieve and existing calibration file or create a new one if it
     * does not exist.
     * @param file path to existing or desired calibration file.
     * @param key key specifying field within calibration file to be modified
     * (e.g. homography).
     * @return TOML table contain
     */
    cpptoml::table generateCalibrationTable(const std::string& file , const std::string& key);
    void saveCalibrationTable(const cpptoml::table& table, const std::string& file);

};

#endif // SAVER_H
