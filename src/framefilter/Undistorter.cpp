//******************************************************************************
//* File:   Undistorter.cpp
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

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <cpptoml.h>

#include "../../lib/utility/OatTOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

#include "Undistorter.h"

namespace oat {

Undistorter::Undistorter(const std::string& source_name, const std::string& sink_name) :
  FrameFilter(source_name, sink_name)
{
    // Nothing
}

void Undistorter::configure(const std::string &config_file,
                            const std::string &config_key) {

    // Available options
    std::vector<std::string> options {"camera-model",
                                      "camera-matrix",
                                      "distortion-coeffs",
                                      "rotation" };

    // This will throw cpptoml::parse_exception if a file
    // with invalid TOML is provided
    auto config = cpptoml::parse_file(config_file);

    // See if a camera configuration was provided
    if (config->contains(config_key)) {

        // Get this components configuration table
        auto this_config = config->get_table(config_key);

        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);

        int64_t val;
        if (oat::config::getValue(this_config,
                                  "camera-model",
                                  val,
                                  static_cast<int64_t>(CameraModel::PINHOLE),
                                  static_cast<int64_t>(CameraModel::FISHEYE))) {

            camera_model_ = static_cast<CameraModel>(val);
        }


        oat::config::Array dc_array;
        if (oat::config::getArray(this_config, "distortion-coeffs", dc_array, true)) {

            auto dc_vec = dc_array->array_of<double>();

            distortion_coefficients_.clear();
            for (auto &dc : dc_vec) {
                distortion_coefficients_.push_back(dc.get()->get());
            }
        }

        // Camera Matrix
        oat::config::Array cal_array;
        if (oat::config::getArray(this_config, "camera-matrix", cal_array, 9, true)) {

            auto camera_vec = cal_array->array_of<double>();

            camera_matrix_(0, 0) = camera_vec[0]->get();
            camera_matrix_(0, 1) = camera_vec[1]->get();
            camera_matrix_(0, 2) = camera_vec[2]->get();
            camera_matrix_(1, 0) = camera_vec[3]->get();
            camera_matrix_(1, 1) = camera_vec[4]->get();
            camera_matrix_(1, 2) = camera_vec[5]->get();
            camera_matrix_(2, 0) = camera_vec[6]->get();
            camera_matrix_(2, 1) = camera_vec[7]->get();
            camera_matrix_(2, 2) = camera_vec[8]->get();

        }

        oat::config::getValue(this_config, "rotation", rotation_deg_, 0.0, 360.0);

    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

void Undistorter::filter(cv::Mat& frame) {

    cv::Mat temp = frame.clone();

    switch (camera_model_) {
        case CameraModel::PINHOLE :
        {
            cv::undistort(temp, frame, camera_matrix_, distortion_coefficients_);
            break;
        }
        case CameraModel::FISHEYE :
        {
            cv::fisheye::undistortImage(temp, frame, camera_matrix_,
                    distortion_coefficients_, cv::Matx33d::eye());
            break;
        }
        default :
        {
            throw std::runtime_error("Invalid camera model selection.\n");
        }
    }

    if (rotation_deg_ != 0.0) {
        cv::Point center = cv::Point(frame.cols/2, frame.rows/2 );
        rotation_matrix_ = cv::getRotationMatrix2D(center, rotation_deg_, 1.0);
        cv::warpAffine(frame, frame, rotation_matrix_, frame.size());
    }

}

} /* namespace oat */
