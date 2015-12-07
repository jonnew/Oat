//******************************************************************************
//* File:   PathChanger.cpp
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

#include <stdexcept>
#include <wordexp.h>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/IOUtility.h"

#include "Calibrator.h"
#include "CameraCalibrator.h"
#include "HomographyGenerator.h"
#include "PathChanger.h"

namespace oat {

void PathChanger::visit(CameraCalibrator* cc) {

    std::cout << "Type the path to save camera calibration information and press <enter>: ";
    try {
        setNewPath(cc);
    } catch (const std::runtime_error& ex) {
        std::cerr << oat::Error(ex.what()) << "\n";
        return;
    }

    std::cout << "Type the key name of this calibration entry and press <enter>: ";
    try {
        setNewKey(cc);
    } catch (const std::runtime_error& ex) {
        std::cerr << oat::Error(ex.what()) << "\n";
        return;
    }


    std::cout << "Camera calibration save file set to " + cc->calibration_save_path() + "\n";
}

void PathChanger::visit(HomographyGenerator* hg) {

    std::cout << "Type the path to save homography information and press <enter>: ";
    try {
        setNewPath(hg);
    } catch (const std::runtime_error& ex) {
        std::cerr << oat::Error(ex.what()) << "\n";
        return;
    }

    std::cout << "Type the key name of this homography entry and press <enter>: ";
    try {
        setNewKey(hg);
    } catch (const std::runtime_error& ex) {
        std::cerr << oat::Error(ex.what()) << "\n";
        return;
    }
    std::cout << "Homography save file set to " + hg->calibration_save_path() + "\n";
}

void PathChanger::setNewPath(Calibrator* cal) {

    std::string new_path;

    if (!std::getline(std::cin, new_path)) {

        // Flush cin in case the uer just inserted crap
        oat::ignoreLine(std::cin);

        throw(std::runtime_error("Invalid input."));
    }

    // Expand the user specified path
    wordexp_t exp_result;
    if (wordexp(new_path.c_str(), &exp_result, WRDE_NOCMD) == 0) {

        new_path = std::string { exp_result.we_wordv[0] };
        wordfree(&exp_result);
        cal->generateSavePath(new_path);

    } else {
        throw std::runtime_error("Could not parse path.");
    }
}

void PathChanger::setNewKey(Calibrator* cal) {

    std::string new_key;

    if (!std::getline(std::cin, new_key)) {

        // Flush cin in case the uer just inserted crap
        oat::ignoreLine(std::cin);

        throw(std::runtime_error("Invalid input."));
    }

    cal->set_calibration_key(new_key);

}

} /* namespace oat */
