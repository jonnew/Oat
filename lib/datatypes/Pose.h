//******************************************************************************
//* File:   Pose.h
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

#ifndef OAT_POSE_H
#define	OAT_POSE_H

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "Sample.h"

namespace oat {

/**
 * @brief Object pose.
 */
class Pose {

friend std::ostream &operator<<(std::ostream &os, const Pose &p);

public:
    enum class DistanceUnit {
        Pixels = 0, //!< Position measured in pixels. Origin is upper left.
        Meters,     //!< Position measured in units specified via homography
    };

    /**
     * Unit of length used to specify rotation and translation vectors.
     */
    DistanceUnit unit_of_length {DistanceUnit::Pixels};

    // Categorical position label (e.g. "North West")
    bool region_valid {false};
    char region[100] {0};

    // Was position detection successful?
    bool found {false};

    void set_sample(const Sample &val) { sample_ = val; }

    // To understand type signatures of these function templates:
    // http://stackoverflow.com/questions/11703553/template-specialization-not-used

    // Orientation getter/setter
    template <typename T>
    void set_orientation(const T &r)
    { (void)r; static_assert(sizeof(T) == 0, "Invalid call to set_orientation()"); }

    template <typename T>
    T orientation() const
    { static_assert(sizeof(T) == 0, "Invalid call to position()"); }

    // Position getter/setter
    template <typename T>
    void set_position(const T &p)
    { (void)p; static_assert(sizeof(T) == 0, "Invalid call to set_position()()"); }

    template <typename T>
    T position() const
    { static_assert(sizeof(T) == 0, "Invalid call to position()"); }

    /**
     * @brief JSON Serializer
     * @param writer Writer to use for serialization
     * @param verbose If true, specifies that fields be serialized even though
     * they contain indeterminate data? This is useful for ease of sample
     * alignment during post processing of saved files.
     * @note Precision template parameter defaults to 3 indicating 0.001 pixels
     * or 1 mm precision
     */
    template <typename Writer, size_t Precision = 3>
    void Serialize(Writer &writer, bool verbose = false) const
    {
        writer.SetMaxDecimalPlaces(Precision);

        writer.StartObject();

        // Sample number
        writer.String("tick");
        writer.Uint64(sample_.count());

        writer.String("usec");
        writer.Uint64(sample_.microseconds().count());

        // Coordinate system
        writer.String("unit");
        writer.Int(static_cast<int>(unit_of_length));

        // Has the object been found?
        writer.String("found");
        writer.Bool(found);

        if (found || verbose) {
            writer.String("position");
            writer.StartArray();
            writer.Double(position_[0]);
            writer.Double(position_[1]);
            writer.Double(position_[2]);
            writer.EndArray(3);
        }

        if (found || verbose) {
            writer.String("orientation");
            writer.StartArray();
            writer.Double(orientation_[0]);
            writer.Double(orientation_[1]);
            writer.Double(orientation_[2]);
            writer.Double(orientation_[3]);
            writer.EndArray(3);
        }

        // Region
        writer.String("reg_ok");
        writer.Bool(region_valid);

        if (region_valid || verbose) {
            writer.String("reg");
            writer.String(region);
        }

        writer.EndObject();
    }

protected:
    /**
     * @brief Current time sample.
     */
    oat::Sample sample_;

    /**
     * @brief Orientation, in units of unit_of_length, represented
     * as a Qauarterion.
     */
    std::array<double, 4> orientation_{{0, 0, 0, 1}};

    /**
     * @brief Position (translation), in units of unit_of_length, away from an
     * external reference frame. This might be the Camera's reference frame or
     * some external reference frame if a homographic transform is used.
     */
    std::array<double, 3> position_{{0, 0, 0}};

    // TODO: Should these travel with pose? Might make downstream visualization much easier
    //std::array<double, 9> camera_matrix_{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    //std::array<double, 8> distortion_coefficients_{{0, 0, 0, 0, 0, 0, 0, 0}};

    // TODO: conditional compilation
    // Map orientation to Eigen::Qauarteriond if available
    Eigen::Map<Eigen::Quaterniond> eig_orient_{orientation_.data()};
    Eigen::Map<Eigen::RowVector3d> eig_pos_{position_.data()};
};

//** SETTERS **//

template <>
inline void Pose::set_position(const std::array<double, 3> &p)
{
    position_ = p;
}

template <>
inline void Pose::set_position(const Eigen::Vector3d &p)
{
    eig_pos_ = p.transpose();
}

template <>
inline void Pose::set_position(const cv::Vec3d &p)
{
    position_[0] = p(0);
    position_[1] = p(1);
    position_[2] = p(2);
}

template <>
inline void Pose::set_orientation(const std::array<double, 4> &r)
{
    orientation_ = r;
}

template <>
inline void Pose::set_orientation(const Eigen::Matrix3d &r)
{
    eig_orient_ = r; // automatic conversion
}

template <>
inline void Pose::set_orientation(const cv::Vec3d &r)
{
    // For opencv rvec, need to go through rotation matrix
    cv::Matx33d R;
    Eigen::Map<Eigen::Matrix3d, Eigen::RowMajor> R_eig(&R.val[0]);
    cv::Rodrigues(r, R);
    eig_orient_ = R_eig; // automatic conversion
}

//** GETTERS **//

/**
 * @brief Get current position vector.
 * @return Position (translation) vector.
 */
template <>
inline std::array<double, 3> Pose::position() const
{
    return position_;
}

template <>
inline cv::Vec3d Pose::position() const
{
    cv::Vec3d p;
    p(0) = position_[0];
    p(1) = position_[1];
    p(2) = position_[2];
    return p;
}

template <>
inline std::array<double, 4> Pose::orientation() const
{
    return orientation_;
}

template <>
inline cv::Vec3d Pose::orientation() const
{
    // OpenCV's rvec is Euler axis multiplied by the angle value
    cv::Vec3d r{0, 0, 0};
    const double angle = -2 * std::acos(orientation_[3]);
    const double alpha = angle / std::sqrt(1 - std::pow(orientation_[3], 2));
    if (!std::isnan(alpha)) {
        r[0] = alpha * orientation_[0];
        r[1] = alpha * orientation_[1];
        r[2] = alpha * orientation_[2];
    }
    //} else {
    //    r[0] = 0;
    //    r[1] = 0;
    //    r[2] = 0;
    //}

    return r;
}

template <>
inline std::array<double, 3> Pose::orientation() const
{
    // OpenCV's rvec is Euler axis multiplied by the angle value
    std::array<double, 3> r{{0, 0, 0}};
    const double angle = -2 * std::acos(orientation_[3]);
    const double alpha = angle / std::sqrt(1 - std::pow(orientation_[3], 2));
    if (!std::isnan(alpha)) {
        r[0] = alpha * orientation_[0];
        r[1] = alpha * orientation_[1];
        r[2] = alpha * orientation_[2];
    }

    return r;
}

//** HELPERS **//

inline std::ostream &operator<<(std::ostream &os, const Pose &p)
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
       << "Position (" << unit << "): [" << p.position_[0] << " " << p.position_[1]
       << " " << p.position_[2] << "]\n"
       << "Orientation: [" << p.orientation_[0] << " " << p.orientation_[1] << " "
       << p.orientation_[2] << " " << p.orientation_[3] << "]";

    return os;
}

}      /* namespace oat */
#endif /* OAT_POSE_H */
