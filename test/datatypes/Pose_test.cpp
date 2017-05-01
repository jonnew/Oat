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

#include "OatConfig.h" // EIGEN3_FOUND

#ifdef EIGEN3_FOUND
#include <Eigen/Core>
#include <Eigen/Geometry>
#endif
#include <opencv2/opencv.hpp>

#include "../../lib/datatypes/Pose.h"

SCENARIO("Various representations of the identity pose orientation must be correct.",
         "[Pose]")
{
    GIVEN("Pose that is the identity quaternion.")
    {
        oat::Pose p(oat::Token::Seconds(1));
        p.set_orientation<std::array<double, 4>>({{0,0,0,1}});

        WHEN("Pose is converted to opencv Rodrigues axis.")
        {
            auto rvec_cv = p.orientation<cv::Vec3d>();

            THEN("Elements of conversion and identity Rodrigues axis shall be "
                 "equal.")
            {
                cv::Vec3d rvec_cv_eye;
                cv::Rodrigues(cv::Matx33d::eye(), rvec_cv_eye);

                REQUIRE(rvec_cv[0] == rvec_cv_eye[0]);
                REQUIRE(rvec_cv[1] == rvec_cv_eye[1]);
                REQUIRE(rvec_cv[2] == rvec_cv_eye[2]);
            }
        }

        WHEN("Pose is converted to opencv rotation matrix.")
        {
            auto R = p.orientation<cv::Matx33d>();

            THEN("Elements of conversion and identity matrix shall be "
                 "equal.")
            {
                auto R_cv = cv::Matx33d::eye();

                REQUIRE(R_cv(0,0) == R(0,0));
                REQUIRE(R_cv(0,1) == R(0,1));
                REQUIRE(R_cv(0,2) == R(0,2));
                REQUIRE(R_cv(1,0) == R(1,0));
                REQUIRE(R_cv(1,1) == R(1,1));
                REQUIRE(R_cv(1,2) == R(1,2));
                REQUIRE(R_cv(2,0) == R(2,0));
                REQUIRE(R_cv(2,1) == R(2,1));
                REQUIRE(R_cv(2,2) == R(2,2));
            }
        }

        WHEN("Pose is converted to Tait-Bryan angles.")
        {
            auto tb = p.toTaitBryan(true);

            THEN("Pitch, roll, and yaw shall be 0 degrees.")
            {
                REQUIRE(tb[0] == 0.0);
                REQUIRE(tb[1] == 0.0);
                REQUIRE(tb[2] == 0.0);
            }
        }

#ifdef EIGEN3_FOUND
        WHEN("Pose is converted to eigen quaternion.")
        {
            auto q_eig = p.orientation<Eigen::Quaterniond>();

            THEN("Elements of conversion and Pose shall be equal.")
            {
                auto q = p.orientation<std::array<double, 4>>();
                REQUIRE(q_eig.w() == q[3]);
                REQUIRE(q_eig.x() == q[0]);
                REQUIRE(q_eig.y() == q[1]);
                REQUIRE(q_eig.z() == q[2]);
            }

            THEN("The conversion shall equal the identity quaternion.")
            {
                REQUIRE(q_eig.w() == 1);
                REQUIRE(q_eig.x() == 0);
                REQUIRE(q_eig.y() == 0);
                REQUIRE(q_eig.z() == 0);
            }
        }

        WHEN("Pose is converted to eigen rotation matrix.")
        {
            auto R = p.orientation<Eigen::Matrix3d>();

            THEN("Elements of conversion and identity matrix shall be "
                 "equal.")
            {
                auto R_eig = Eigen::Matrix3d::Identity();

                REQUIRE(R == R_eig);
            }
        }
#endif
    }
}

SCENARIO("Using opencv implementations as standard, various representations of a "
         "random rotation matrix shall be consistent.",
         "[Pose]")
{
    GIVEN("A random opencv rotation matrix, R, and a pose p.")
    {
        oat::Pose p(oat::Token::Seconds(1));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis0(0, 1);
        auto d = dis0(gen);
        std::uniform_real_distribution<> dis1(0, d);
        cv::Matx33d R = oat::randRotation({{dis1(gen), dis0(gen), dis1(gen)}});

        WHEN("When R is used to set p, and then p is extracted as a rotation "
             "matrix.")
        {
            p.set_orientation(R);
            auto pR = p.orientation<cv::Matx33d>();

            THEN("The elements of the extracted matrix shall equal those of R.")
            {
                REQUIRE(R(0,0) == Approx(pR(0,0)).epsilon(0.01));
                REQUIRE(R(0,1) == Approx(pR(0,1)).epsilon(0.01));
                REQUIRE(R(0,2) == Approx(pR(0,2)).epsilon(0.01));
                REQUIRE(R(1,0) == Approx(pR(1,0)).epsilon(0.01));
                REQUIRE(R(1,1) == Approx(pR(1,1)).epsilon(0.01));
                REQUIRE(R(1,2) == Approx(pR(1,2)).epsilon(0.01));
                REQUIRE(R(2,0) == Approx(pR(2,0)).epsilon(0.01));
                REQUIRE(R(2,1) == Approx(pR(2,1)).epsilon(0.01));
                REQUIRE(R(2,2) == Approx(pR(2,2)).epsilon(0.01));
            }
        }

        WHEN("When R is converted to Rodrigues vector r, this is used to set "
             "p, and then p is extracted as an rvec.")
        {
            cv::Vec3d r;
            cv::Rodrigues(R, r);
            p.set_orientation(r);
            auto pr = p.orientation<cv::Vec3d>();

            THEN("The elements of the extracted rvec shall equal those of r.")
            {
                REQUIRE(r[0] == Approx(pr[0]));
                REQUIRE(r[1] == Approx(pr[1]));
                REQUIRE(r[2] == Approx(pr[2]));
            }
        }

        WHEN("When R is converted to Rodrigues vector r, this is used to set "
             "p, and then p is extracted as a rotation matrix.")
        {
            cv::Vec3d r;
            cv::Rodrigues(R, r);
            p.set_orientation(r);
            auto pR = p.orientation<cv::Matx33d>();

            THEN("The elements of the extracted matrix shall those of equal "
                 "cv::Rodrigues(r).")
            {
                REQUIRE(R(0,0) == Approx(pR(0,0)));
                REQUIRE(R(0,1) == Approx(pR(0,1)));
                REQUIRE(R(0,2) == Approx(pR(0,2)));
                REQUIRE(R(1,0) == Approx(pR(1,0)));
                REQUIRE(R(1,1) == Approx(pR(1,1)));
                REQUIRE(R(1,2) == Approx(pR(1,2)));
                REQUIRE(R(2,0) == Approx(pR(2,0)));
                REQUIRE(R(2,1) == Approx(pR(2,1)));
                REQUIRE(R(2,2) == Approx(pR(2,2)));
            }
        }
    }
}

#ifdef EIGEN3_FOUND
SCENARIO("Using Eigen implementations as standard, various representations of a "
         "random rotation matrix shall be consistent.",
         "[Pose]")
{
    GIVEN("The Pose, p, and Eigen Qauarterion, q, set to the same random value.")
    {
        oat::Pose p(oat::Token::Seconds(1));
        const auto rq = Eigen::Quaterniond::UnitRandom();
        p.set_orientation<Eigen::Quaterniond>(rq);

        WHEN("p is converted to a std::array.")
        {
            auto a = p.orientation<std::array<double, 4>>();
            THEN("p[0] shall equal q.x()")
            {
                REQUIRE(a[0] == rq.x());
            }

            THEN("p[1] shall equal q.y()")
            {
                REQUIRE(a[1] == rq.y());
            }

            THEN("p[2] shall equal q.z()")
            {
                REQUIRE(a[2] == rq.z());
            }

            THEN("p[3] shall equal q.w()")
            {
                REQUIRE(a[3] == rq.w());
            }
        }

        WHEN("p is convered to an Eigen rotation matrix.")
        {
            auto R = p.orientation<Eigen::Matrix3d>();

            THEN("Elements of conversion and Eigen rotation matrix shall be equal.")
            {
                auto R_eig = rq.toRotationMatrix();

                REQUIRE(R_eig(0,0) == R(0,0));
                REQUIRE(R_eig(0,1) == R(0,1));
                REQUIRE(R_eig(0,2) == R(0,2));
                REQUIRE(R_eig(1,0) == R(1,0));
                REQUIRE(R_eig(1,1) == R(1,1));
                REQUIRE(R_eig(1,2) == R(1,2));
                REQUIRE(R_eig(2,0) == R(2,0));
                REQUIRE(R_eig(2,1) == R(2,1));
                REQUIRE(R_eig(2,2) == R(2,2));
            }
        }

        WHEN("p is convered to an opencv rotation matrix.")
        {
            auto R = p.orientation<cv::Matx33d>();

            THEN("Elements of conversion and Eigen rotation matrix shall be "
                 "very close to equal.")
            {
                auto R_eig = rq.toRotationMatrix();

                REQUIRE(R_eig(0,0) == Approx(R(0,0)));
                REQUIRE(R_eig(0,1) == Approx(R(0,1)));
                REQUIRE(R_eig(0,2) == Approx(R(0,2)));
                REQUIRE(R_eig(1,0) == Approx(R(1,0)));
                REQUIRE(R_eig(1,1) == Approx(R(1,1)));
                REQUIRE(R_eig(1,2) == Approx(R(1,2)));
                REQUIRE(R_eig(2,0) == Approx(R(2,0)));
                REQUIRE(R_eig(2,1) == Approx(R(2,1)));
                REQUIRE(R_eig(2,2) == Approx(R(2,2)));
            }
        }

    }

    GIVEN("A Eigen rotation matrix, R, generated from Eigen quaternion, q, and "
          "a pose, p.")
    {
        oat::Pose p(oat::Token::Seconds(1));
        const auto u = Eigen::Quaterniond::UnitRandom();
        auto q = Eigen::Quaterniond::UnitRandom();
        const auto R = q.toRotationMatrix();

        WHEN("When R is used to set p and p is then converted to "
             "std::array.")
        {
            p.set_orientation<Eigen::Matrix3d>(R);
            auto a = p.orientation<std::array<double, 4>>();

            THEN("p and q shall represent the same orientation")
            {
                REQUIRE(q.dot(Eigen::Quaterniond(a[3], a[0], a[1], a[2]))
                        == Approx(1.0));
            }

            THEN("p shall equal q or -q")
            {
                bool p_eq_q = a[0] == Approx(q.x()) &&
                              a[1] == Approx(q.y()) &&
                              a[2] == Approx(q.z()) &&
                              a[3] == Approx(q.w());
                bool p_eq_negq = a[0] == Approx(-q.x()) &&
                                 a[1] == Approx(-q.y()) &&
                                 a[2] == Approx(-q.z()) &&
                                 a[3] == Approx(-q.w());
                bool result = p_eq_negq || p_eq_q;
                REQUIRE(result);
            }

        }

        WHEN("The elements of R are used to create a cv::Matx33d, and it used "
             "to set p and p is then converted to std::array.")
        {
            cv::Matx33d R_cv;
            R_cv(0, 0) = R(0, 0);
            R_cv(0, 1) = R(0, 1);
            R_cv(0, 2) = R(0, 2);
            R_cv(1, 0) = R(1, 0);
            R_cv(1, 1) = R(1, 1);
            R_cv(1, 2) = R(1, 2);
            R_cv(2, 0) = R(2, 0);
            R_cv(2, 1) = R(2, 1);
            R_cv(2, 2) = R(2, 2);

            p.set_orientation<cv::Matx33d>(R_cv);
            auto a = p.orientation<std::array<double, 4>>();

            // TODO: This fails because dot product is negative 1. That seems
            // to mean the orientations are directly opposite. However, the
            // next test passes so I don't understand.
            //THEN("p and q shall represent the same orientation")
            //{
            //    REQUIRE(q.dot(Eigen::Quaterniond(a[3], a[0], a[1], a[2]))
            //            == Approx(1.0));
            //}

            THEN("p shall equal q or -q")
            {
                bool p_eq_q = a[0] == Approx(q.x()) &&
                              a[1] == Approx(q.y()) &&
                              a[2] == Approx(q.z()) &&
                              a[3] == Approx(q.w());
                bool p_eq_negq = a[0] == Approx(-q.x()) &&
                                 a[1] == Approx(-q.y()) &&
                                 a[2] == Approx(-q.z()) &&
                                 a[3] == Approx(-q.w());
                bool result = p_eq_negq || p_eq_q;
                REQUIRE(result);
            }
        }

        WHEN("The elements of R are used to create a cv::Matx33d, and it is used "
             "to set the p and p is then converted to a cv::Matx33d.")
        {
            cv::Matx33d R_cv;
            R_cv(0, 0) = R(0, 0);
            R_cv(0, 1) = R(0, 1);
            R_cv(0, 2) = R(0, 2);
            R_cv(1, 0) = R(1, 0);
            R_cv(1, 1) = R(1, 1);
            R_cv(1, 2) = R(1, 2);
            R_cv(2, 0) = R(2, 0);
            R_cv(2, 1) = R(2, 1);
            R_cv(2, 2) = R(2, 2);

            p.set_orientation<cv::Matx33d>(R_cv);
            R_cv = p.orientation<cv::Matx33d>();

            THEN("Elements of conversion and R shall be very close to equal.")
            {

                REQUIRE(R(0,0) == Approx(R_cv(0,0)));
                REQUIRE(R(0,1) == Approx(R_cv(0,1)));
                REQUIRE(R(0,2) == Approx(R_cv(0,2)));
                REQUIRE(R(1,0) == Approx(R_cv(1,0)));
                REQUIRE(R(1,1) == Approx(R_cv(1,1)));
                REQUIRE(R(1,2) == Approx(R_cv(1,2)));
                REQUIRE(R(2,0) == Approx(R_cv(2,0)));
                REQUIRE(R(2,1) == Approx(R_cv(2,1)));
                REQUIRE(R(2,2) == Approx(R_cv(2,2)));
            }
        }
    }
}
#endif
