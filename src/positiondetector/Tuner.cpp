//******************************************************************************
//* File:   Tuner.cpp
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

#include <iostream>
#include <iomanip>
#include <sstream>

#include <opencv2/highgui.hpp>

#include "Tuner.h"

namespace oat {

Tuner::Tuner(const std::string &window_name)
: w_(window_name)
{
#ifdef HAVE_OPENGL // TODO: replace with CV-specific opengl and get rid of
                   // try-catch
    try {
        cv::namedWindow(w_, cv::WINDOW_OPENGL & cv::WINDOW_KEEPRATIO);
    } catch (cv::Exception &ex) {
        whoWarn(w_,
                "OpenCV not compiled with OpenGL support. Falling "
                "back to OpenCV's display driver.\n");
        cv::namedWindow(w_, cv::WINDOW_NORMAL & cv::WINDOW_KEEPRATIO);
    }
#else
    cv::namedWindow(w_, cv::WINDOW_NORMAL);
#endif
}

Tuner::~Tuner()
{
    for (auto &p : cb_params_)
        delete static_cast<CBBase *>(p);
}

void Tuner::tune(const oat::Frame &frame,
                 const oat::Pose &pose,
                 const cv::Matx33d &K = cv::Matx33d::eye(),
                 const std::vector<double> &D = {0, 0, 0, 0, 0, 0, 0, 0})
{
    // Make sure this is color image
    oat::Frame img = frame;
    oat::convertColor(frame, img, PIX_BGR);

    // Messages to be printed on screen
    std::vector<std::string> msgs;

    if (pose.found ) {

        double length = 0.0;
        if (pose.unit_of_length == Pose::DistanceUnit::Pixels)
            length = 50; // TODO: This is brittle
        else if (pose.unit_of_length == Pose::DistanceUnit::Meters)
            length = 0.1; // TODO: This is brittle

        // Make axis
        std::vector<cv::Point3f> axis_3d;
        axis_3d.push_back(cv::Point3f(0, 0, 0));
        axis_3d.push_back(cv::Point3f(length, 0, 0));
        axis_3d.push_back(cv::Point3f(0, length, 0));
        axis_3d.push_back(cv::Point3f(0, 0, length));
        std::vector<cv::Point2f> axis_2d;
        cv::projectPoints(axis_3d,
                          pose.orientation<cv::Vec3d>(),
                          pose.position<cv::Vec3d>(),
                          K,
                          D,
                          axis_2d);

        // Draw axis or point
        if (pose.orientation_dof >= Pose::DOF::Two) {
            cv::line(img, axis_2d[0], axis_2d[1], cv::Scalar(0, 0, 255), 3);
            cv::line(img, axis_2d[0], axis_2d[2], cv::Scalar(0, 255, 0), 3);
            if (pose.orientation_dof == Pose::DOF::Three)
                cv::line(img, axis_2d[0], axis_2d[3], cv::Scalar(255, 0, 0), 3);
        } else {
            cv::circle(img, axis_2d[0], length / 2, cv::Scalar(0, 255, 255), 3);
        }

        // Position
        auto p = pose.position<std::array<double, 3>>();
        std::stringstream m;
        m << std::setprecision(2) << std::fixed;
        if (pose.unit_of_length == Pose::DistanceUnit::Meters)
            m << "P (m): [";
        else if (pose.unit_of_length == Pose::DistanceUnit::Pixels)
            m << "P (px): [";
        m << p[0] << ", " << p[1] << ", " << p[2] << "]";
        msgs.push_back(m.str());

        // Tait-bryan angles
        auto tb = pose.toTaitBryan(true);
        m.str("");
        m << " (deg): [" << tb[0] << ", " << tb[1] << ", " << tb[2] << "]";
        msgs.push_back(m.str());

    } else {
        msgs.push_back("Not found");
    }

    for (std::vector<std::string>::size_type i = 0; i < msgs.size(); i++) {

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(msgs[i], 1, 1, 1, &baseline);
        cv::Point text_origin(frame.cols - textSize.width - 10,
                              frame.rows - (i * (textSize.height + 10)) - 10);
        cv::putText(frame, msgs[i], text_origin, 1, 1, cv::Scalar(0, 255, 255));
    }

    cv::imshow(w_, img);
    cv::waitKey(1);
}

} /* namespace oat */
