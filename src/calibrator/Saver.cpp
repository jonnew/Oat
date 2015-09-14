//******************************************************************************
//* File:   Saver.cpp
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

#include <memory>
#include <stdexcept>

#include <opencv2/core/mat.hpp>

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/IOUtility.h"

#include "CameraCalibrator.h"
#include "HomographyGenerator.h"
#include "Saver.h"

void Saver::visit(CameraCalibrator* camera_calibrator) {

}

void Saver::visit(HomographyGenerator* hg) {

    // Check the the homography is valid
    if (!hg->homography_valid()) {
        std::cerr << oat::Error("Homgraphy must be computed before it is saved.\n");
        hg->printUsage(std::cout);
        return;
    }

    // Generate or get base calibration table
    cpptoml::table calibration;
    try {
        calibration = generateCalibrationTable(hg->calibration_save_path(), "homography");
    } catch (const std::runtime_error& ex) {
        std::cerr << ex.what();
        return;
    }
    
    auto dt = std::make_shared<cpptoml::value<cpptoml::datetime>>(generateDateTime()); 
    calibration.insert("last-modified", dt);

    // Construct TOML array from homography
    auto arr = std::make_shared<cpptoml::array>();
    cv::MatConstIterator_<double> it, end;
    for (it = hg->homography().begin<double>(), end = hg->homography().end<double>(); it != end; ++it)  {
        arr->get().push_back(std::make_shared<cpptoml::value<double>>(*it));
    }

    // Insert the array into the calibration table
    calibration.insert("homography", arr);

    // Save the file
    saveCalibrationTable(calibration, hg->calibration_save_path());
}

cpptoml::table Saver::generateCalibrationTable(const std::string& file, const std::string& key) {

    cpptoml::table table;

    // If the file already exists, open it as a TOML table
    try {
        table = cpptoml::parse_file(file);
    } catch (const cpptoml::parse_exception& ex) {
        // Do nothing because we will just overwrite.
        // TODO: This is dangerous. What if a user made a table themselves that
        // has a slight syntax error. This will just destroy their work
    }

    // See if there is already specified key. If so warn the user that it
    // will be overwritten by proceeding
    if (table.contains(key)) {

        std::cout << file + " already contains a " + key + " entry. Overwrite? (y/n): ";

        char yes;
        if (!(std::cin >> yes) || (yes != 'y' && yes != 'Y')) {

            // Flush cin in case the uer just inserted crap
            oat::ignoreLine(std::cin);

            // Needs to be caught by caller
            throw(std::runtime_error("Save aborted.\n"));
        }
    }
}

cpptoml::datetime Saver::generateDateTime() {

    // Generate current date-time 
    std::time_t raw_time;
    struct tm * time_info;
    char buffer[100];
    std::time(&raw_time);
    time_info = std::localtime(&raw_time);

    cpptoml::datetime dt;
    dt.year = time_info->tm_year + 1900;
    dt.month = time_info->tm_mon + 1;
    dt.day = time_info->tm_mday;
    dt.hour = time_info->tm_hour;
    dt.minute = time_info->tm_min;
    dt.second = time_info->tm_sec;

    return dt;
}

void Saver::saveCalibrationTable(const cpptoml::table& table, const std::string& file) {

    // Save the table as TOML file
    std::ofstream fs;
    fs.exceptions (std::ofstream::failbit | std::ofstream::badbit);
    try {
        fs.open(file, std::ios::out);
        fs << table;
        std::cout << "Calibration saved to " + file + "\n";
        fs.close();
    } catch (std::ofstream::failure& ex) {
        std::cerr << oat::Error("Could not write to " + file + ".\n");
    }
}
