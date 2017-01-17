//******************************************************************************
//* File:   Pose_test.cpp
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

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "../../lib/datatypes/Pose.h"

SCENARIO("Angle axis rotation must be the same across OpenCV, Eigen, and my "
         "implementations.",
         "[Pose]")
{
    GIVEN("A the identity quaternion, Rodrigues axis, and Rotation Matrix.")
    {
        oat::Pose p;
        p.set_orientation<std::array<double, 4>>({{0,0,0,1}});
        cv::Matx33d R_cv = cv::Matx33d::eye();
        cv::Vec3d rvec_cv(0, 0, 0);

        // TODO
        //auto R_eig = Eigen::Matrix<double, 3, 3>::Identity();

        WHEN("Pose is converted to Rodrigues axis.")
        {
            auto rvec = p.orientation<cv::Vec3d>();

            THEN("Elements of conversion and identity Rodrigues axis shall be "
                 "equal.")
            {
                REQUIRE(rvec[0] == rvec_cv[0]);
                REQUIRE(rvec[1] == rvec_cv[1]);
                REQUIRE(rvec[2] == rvec_cv[2]);
            }
        }
    }
}

// TODO:
//auto q_eig = Eigen::Quaterniond(R_eig);
//auto r_eig = Eigen::AngleAxisd(R_eig);
//auto R_cv = cv::Matx33d::eye();
//cv::Mat r_cv;
//cv::Rodrigues(R_cv, r_cv);
//
//cout << "Eigen Rotation Matrix:\n";
//cout << R_eig << endl;
//
//cout << "Eigen Quaternion:\n";
//cout << "[" << q_eig.x() << " " << q_eig.y() << " " << q_eig.z()
//     << " " << q_eig.w() << "]" << endl;
//
//cout << "Eigen rvec\n";
//cout << r_eig.axis() << endl;
//
//cout << "CV Rotation Matrix:\n";
//cout << R_cv << endl;
//
//cout << "CV Quaternion:\n";
//cout << "NA";
//
//cout << "CV rvec\n";
//cout << r_cv << endl;
//
//cout << "** Random Matrix **\n\n";
//
//q_eig = Eigen::Quaterniond::UnitRandom();
//auto R_eig1 = q_eig.toRotationMatrix();
//r_eig = Eigen::AngleAxisd(R_eig);
//R_cv = cv::Matx33d{R_eig1(0),
//                   R_eig1(1),
//                   R_eig1(2),
//                   R_eig1(3),
//                   R_eig1(4),
//                   R_eig1(5),
//                   R_eig1(6),
//                   R_eig1(7),
//                   R_eig1(8)};
//cv::Rodrigues(R_cv, r_cv);
//
//cout << "Eigen Rotation Matrix:\n";
//cout << R_eig1 << endl;
//
//cout << "Eigen Quaternion:\n";
//cout << "[" << q_eig.x() << " " << q_eig.y() << " " << q_eig.z()
//     << " " << q_eig.w() << "]" << endl;
//
//cout << "Eigen rvec\n";
//cout << r_eig.axis() << endl;
//
//cout << "CV Rotation Matrix:\n";
//cout << R_cv << endl;
//
//cout << "CV Quaternion:\n";
//cout << "NA";
//
//cout << "CV rvec\n";
//cout << r_cv << endl;

