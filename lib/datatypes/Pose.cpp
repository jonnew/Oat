//******************************************************************************
//* File:   Pose.cpp
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

#include "Pose.h"

namespace oat {

std::ostream &operator<<(std::ostream &os, const Pose &p)
{
    const char *unit;
    switch (p.unit_of_length) {
        case Pose::DistanceUnit::Pixels:
            unit = "px";
            break;
        case Pose::DistanceUnit::Meters:
            unit = "m";
            break;
    }

    os << "Time (s): " << p.sample_.seconds().count() << "\n"
       << "Region: " << p.region << "\n"
       << "Position (" << unit << "): ["
       << p.p_[0] << " " << p.p_[1] << " " << p.p_[2] << "]\n"
       << "Orientation: ["
       << p.q_[w] << " " << p.q_[x] << " " << p.q_[y] << " " << p.q_[z] << "]";

    return os;
}

std::vector<char> packPose(const Pose &p)
{
    std::vector<char> pack;
    pack.reserve(oat::POSE_NPY_DTYPE_BYTES);

    auto sc = p.sample_.count();
    auto val = reinterpret_cast<char *>(&sc);
    pack.insert(pack.end(), val, val + sizeof(sc));

    auto su = p.sample_.microseconds().count();
    val = reinterpret_cast<char *>(&su);
    pack.insert(pack.end(), val, val + sizeof(su));

    auto u = static_cast<int>(p.unit_of_length);
    val = reinterpret_cast<char *>(&u);
    pack.insert(pack.end(), val, val + sizeof(u));

    // Found?
    char found = p.found ? 1 : 0;
    pack.insert(pack.end(), &found, &found + 1);

    // Position
    for (auto &coord : p.p_) {
        auto tmp = coord;
        val = reinterpret_cast<char *>(&tmp);
        pack.insert(pack.end(), val, val + sizeof(tmp));
    }

    // Orientation
    for (auto &quat : p.q_) {
        auto tmp = quat;
        val = reinterpret_cast<char *>(&tmp);
        pack.insert(pack.end(), val, val + sizeof(tmp));
    }

    // Region
    char rok = p.in_region ? 1 : 0;
    pack.insert(pack.end(), &rok, &rok + 1);
    pack.insert(pack.end(), p.region, p.region + oat::Pose::region_max_char);

    return pack;
}

std::array<double, 3> Pose::toTaitBryan(const bool deg) const
{
    std::array<double, 3> a;

    const auto xx = q_[x] * q_[x];
    const auto yy = q_[y] * q_[y];
    const auto zz = q_[z] * q_[z];

    const auto xy = q_[x] * q_[y];
    const auto xz = q_[x] * q_[z];
    const auto xw = q_[x] * q_[w];

    const auto yz = q_[y] * q_[z];
    const auto yw = q_[y] * q_[w];

    const auto zw = q_[z] * q_[w];

    if ((xy + zw) == 0.5) {
        a[0] = 2 * std::asin(2 * xy + 2 * zw);
        a[1] = std::atan2(q_[x], q_[w]);
        a[2] = 0;
    } else if ((xy + zw) == -0.5) {
        a[0] = -2 * std::atan2(q_[x], q_[w]);
        a[1] = std::atan2(q_[x], q_[w]);
        a[2] = 0;
    } else {
        a[0] = std::atan2(2 * yw - 2 * xz, 1 - 2 * yy - 2 * zz);
        a[1] = std::asin(2 * xy + 2 * zw);
        a[2] = std::atan2(2 * xw - 2 * yz, 1 - 2 * xx - 2 * zz);
    }

    if (deg) {
        a[0] *= 57.2958;
        a[1] *= 57.2958;
        a[2] *= 57.2958;
    }

    return a;
}

cv::Matx33d randRotation(std::array<double, 3> x)
{
    float theta = x[0] * M_PI * 2; // Rotation about the pole (Z).
    float phi = x[1] * M_PI * 2;   // For direction of pole deflection.
    float z = x[2] * 2.0;          // For magnitude of pole deflection.

    // Compute a vector V used for distributing points over the sphere via the
    // reflection I - V Transpose(V).  This formulation of V will guarantee
    // that if x[1] and x[2] are uniformly distributed, the reflected points
    // will be uniform on the sphere.  Note that V has length sqrt(2) to
    // eliminate the 2 in the Householder matrix.

    float r = std::sqrt(z);
    float Vx = std::sin(phi) * r;
    float Vy = std::cos(phi) * r;
    float Vz = std::sqrt(2.0 - z);

    // Compute the row vector S = Transpose(V) * R, where R is a simple
    // rotation by theta about the z-axis.  No need to compute Sz since it's
    // just Vz.

    float st = std::sin(theta);
    float ct = std::cos(theta);
    float Sx = Vx * ct - Vy * st;
    float Sy = Vx * st + Vy * ct;

    // Construct the rotation matrix  ( V Transpose(V) - I ) R, which is
    // equivalent to V S - R.

    cv::Matx33d R;
    R(0, 0) = Vx * Sx - ct;
    R(0, 1) = Vx * Sy - st;
    R(0, 2) = Vx * Vz;

    R(1, 0) = Vy * Sx + st;
    R(1, 1) = Vy * Sy - ct;
    R(1, 2) = Vy * Vz;

    R(2, 0) = Vz * Sx;
    R(2, 1) = Vz * Sy;
    R(2, 2) = 1.0 - z; // This equals Vz * Vz - 1.0

    return R;
}

} /* namespace oat */
