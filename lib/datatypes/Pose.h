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

#include "OatConfig.h" // EIGEN3_FOUND

#include <vector>
#include <cstring>

#include <cmath>
#include <opencv2/calib3d.hpp>
#include <rapidjson/prettywriter.h>
#ifdef EIGEN3_FOUND
#include <Eigen/Core>
#include <Eigen/Geometry>
#endif

#include "Sample.h"

/**
 * @def Q2R
 * @note Animating rotation with quaternion curves, Ken Shoemake, SIGGRAPH '85
 * Proceedings of the 12th annual conference on Computer graphics and
 * interactive techniques Pages 245-25
 */
#define Q2R                                                                    \
    const auto xx = q_[x] * q_[x];                                             \
    const auto yy = q_[y] * q_[y];                                             \
    const auto zz = q_[z] * q_[z];                                             \
    const auto xy = q_[x] * q_[y];                                             \
    const auto xz = q_[x] * q_[z];                                             \
    const auto xw = q_[x] * q_[w];                                             \
    const auto yz = q_[y] * q_[z];                                             \
    const auto yw = q_[y] * q_[w];                                             \
    const auto zw = q_[z] * q_[w];                                             \
    R(0, 0) = 1 - 2 * yy - 2 * zz;                                             \
    R(0, 1) = 2 * xy - 2 * zw;                                                 \
    R(0, 2) = 2 * xz + 2 * yw;                                                 \
    R(1, 0) = 2 * xy + 2 * zw;                                                 \
    R(1, 1) = 1 - 2 * xx - 2 * zz;                                             \
    R(1, 2) = 2 * yz - 2 * xw;                                                 \
    R(2, 0) = 2 * xz - 2 * yw;                                                 \
    R(2, 1) = 2 * yz + 2 * xw;                                                 \
    R(2, 2) = 1 - 2 * xx - 2 * yy;

/**
 * @def R2Q
 * @note Animating rotation with quaternion curves, Ken Shoemake, SIGGRAPH '85
 * Proceedings of the 12th annual conference on Computer graphics and
 * interactive techniques Pages 245-25
 */
#define R2Q                                                                    \
    auto t = 0.25 * (1 + R(0, 0) + R(1, 1) + R(2, 2));                         \
    if (t > std::numeric_limits<double>::epsilon()) {                          \
        q_[w] = std::sqrt(t);                                                  \
        t = 0.25 / q_[w];                                                      \
        q_[x] = t * (R(2, 1) - R(1, 2));                                       \
        q_[y] = t * (R(0, 2) - R(2, 0));                                       \
        q_[z] = t * (R(1, 0) - R(0, 1));                                       \
    } else {                                                                   \
        q_[w] = 0;                                                             \
        t = -0.5 * (R(1, 1) + R(2, 2));                                        \
        if (t > std::numeric_limits<double>::epsilon()) {                      \
            q_[x] = std::sqrt(t);                                              \
            t = 0.5 / q_[x];                                                   \
            q_[y] = t * R(0, 1);                                               \
            q_[z] = t * R(0, 3);                                               \
        } else {                                                               \
            q_[x] = 0;                                                         \
            t = 0.5 * (1 - R(2, 2));                                           \
            if (t > std::numeric_limits<double>::epsilon()) {                  \
                q_[y] = std::sqrt(t);                                          \
                q_[z] = R(1, 2) / (2 * q_[y]);                                 \
            } else {                                                           \
                q_[y] = 0;                                                     \
                q_[z] = 1;                                                     \
            }                                                                  \
        }                                                                      \
    }

// Eigen implementaiton (seems to have sign convension issues compared to mine)
//#define R2Q                                                                    \
//    auto t = R(0, 0) + R(1, 1) + R(2, 2);                                      \
//    if (t > 0) {                                                               \
//        t = std::sqrt(t + 1.0);                                                \
//        q_[w] = 0.5 * t;                                                       \
//        t = 0.5 / t;                                                           \
//        q_[x] = R(2, 1) - R(1, 2) * t;   \\ t * (R21 - R12) ??                 \
//        q_[y] = R(0, 2) - R(2, 0) * t;                                         \
//        q_[z] = R(1, 0) - R(0, 1) * t;                                         \
//    } else {                                                                   \
//        size_t i = 0;                                                          \
//        if (R(1, 1) > R(0, 0))                                                 \
//            i = 1;                                                             \
//        if (R(2, 2) > R(i, i))                                                 \
//            i = 2;                                                             \
//        size_t j = (i + 1) % 3;                                                \
//        size_t k = (j + 1) % 3;                                                \
//                                                                               \
//        t = std::sqrt(R(i, i) - R(j, j) - R(k, k) + 1.0);                      \
//        q_[i] = 0.5 * t;                                                       \
//        t = 0.5 / t;                                                           \
//        q_[w] = (R(k, j) - R(j, k)) * t;                                       \
//        q_[j] = (R(j, i) + R(i, j)) * t;                                       \
//        q_[k] = (R(k, i) + R(i, k)) * t;                                       \
//    }

namespace oat {

static constexpr size_t x{0}, y{1}, z{2}, w{3};

// Forward decl.
class Pose;
template <typename Writer, size_t Precision = 3>
void serializePose(const Pose &p, Writer &writer, bool verbose = true);

/**
 * @brief Object pose.
 */
class Pose {

    template <typename Writer, size_t Precision>
    friend void serializePose(const Pose &p, Writer &w, bool verbose);
    friend std::ostream &operator<<(std::ostream &os, const Pose &p);

public:
    enum class DistanceUnit {
        Pixels = 0, //!< Position measured in pixels. Origin is
                    //! upper left.
        Meters,     //!< Position measured in units specified via
                    //! homography
    };

    Pose() = default;
    Pose(const Pose &p)
    : unit_of_length(p.unit_of_length), region_valid(p.region_valid),
      found(p.found), sample_(p.sample_), q_(p.q_), p_(p.p_)
    {
        std::memcpy(region, p.region, sizeof(p.region));
#ifdef EIGEN3_FOUND
        Eigen::Map<Eigen::Quaterniond> eig_orient_{q_.data()};
        Eigen::Map<Eigen::RowVector3d> eig_pos_{p_.data()};
#endif
    }
    Pose(Pose &&p) = default;
    Pose &operator=(const Pose &rhs)
    {
        unit_of_length = rhs.unit_of_length;
        region_valid = rhs.region_valid;
        std::memcpy(region, rhs.region, sizeof(rhs.region));
        found = rhs.found;
        sample_ = rhs.sample_;
        q_ = rhs.q_;
        p_ = rhs.p_;

#ifdef EIGEN3_FOUND
        Eigen::Map<Eigen::Quaterniond> eig_orient_{q_.data()};
        Eigen::Map<Eigen::RowVector3d> eig_pos_{p_.data()};
#endif

        return *this;
    }
    Pose &operator=(Pose &&) = default;

    /**
     * @brief Unit of length used to specify rotation and translation vectors.
     */
    DistanceUnit unit_of_length{DistanceUnit::Pixels};

    // Categorical position label (e.g. "North West")
    bool region_valid{false};
    char region[100]{0};

    // Was position detection successful?
    bool found{false};

    void set_sample(const Sample &val) { sample_ = val; }

    // To understand type signatures of these function
    // templates:
    // http://stackoverflow.com/questions/11703553/template-specialization-not-used

    // Orientation getter/setter
    template <typename T>
    void set_orientation(const T &r)
    {
        (void)r;
        static_assert(sizeof(T) == 0, "Invalid call to set_orientation()");
    }

    template <typename T>
    T orientation() const
    {
        static_assert(sizeof(T) == 0, "Invalid call to position()");
    }

    // Position getter/setter
    template <typename T>
    void set_position(const T &p)
    {
        (void)p;
        static_assert(sizeof(T) == 0, "Invalid call to set_position()()");
    }

    template <typename T>
    T position() const
    {
        static_assert(sizeof(T) == 0, "Invalid call to position()");
    }

    // Get axis angle representation. Cannot use a template
    // specialization
    // straightforwardly because this hs the same return type as
    // rvec.
    std::array<double, 3> toTaitBryan(const bool deg = false) const;

protected:
    /**
     * @brief Current time sample.
     */
    oat::Sample sample_;

    /**
     * @brief Orientation, in units of unit_of_length, represented
     * as a Qauarterion. Initialized to identity.
     * @note Memory layout maps that of Eigen::Quaterniond;
     */
    std::array<double, 4> q_{{0, 0, 0, 1}};

    /**
     * @brief Position (translation), in units of unit_of_length, away from an
     * external reference frame. This might be the Camera's reference frame or
     * some external reference frame if a homographic transform is used.
     */
    std::array<double, 3> p_{{0, 0, 0}};

// TODO: Should these travel with pose? Might make downstream visualization much
// easier
// std::array<double, 9> camera_matrix_{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
// std::array<double, 8> distortion_coefficients_{{0, 0, 0, 0, 0, 0, 0, 0}};

#ifdef EIGEN3_FOUND
    Eigen::Map<Eigen::Quaterniond> eig_orient_{q_.data()};
    Eigen::Map<Eigen::RowVector3d> eig_pos_{p_.data()};
#endif
};

//** POSITION SETTERS **//

template <>
inline void Pose::set_position(const std::array<double, 3> &p)
{
    p_ = p;
}

template <>
inline void Pose::set_position(const cv::Vec3d &p)
{
    p_[0] = p(0);
    p_[1] = p(1);
    p_[2] = p(2);
}

#ifdef EIGEN3_FOUND
template <>
inline void Pose::set_position(const Eigen::Vector3d &p)
{
    eig_pos_ = p.transpose();
}
#endif

//** ORIENTATION SETTERS **//

template <>
inline void Pose::set_orientation(const std::array<double, 4> &r)
{
    q_ = r;
}

template <>
inline void Pose::set_orientation(const cv::Matx33d &R)
{
    R2Q // Macro
}

template <>
inline void Pose::set_orientation(const cv::Vec3d &r)
{
    // Probably some way to go directly to quaternion from rvec,
    // but who cares I
    // guess
    cv::Matx33d R;
    cv::Rodrigues(r, R);
    R2Q // macro
    //r_ = r;
}

#ifdef EIGEN3_FOUND
template <>
inline void Pose::set_orientation(const Eigen::Quaterniond &q)
{
    eig_orient_ = q; // automatic conversion
}

template <>
inline void Pose::set_orientation(const Eigen::Matrix3d &r)
{
    eig_orient_ = r; // automatic conversion
}
#endif

//** POSITION GETTERS **//

/**
 * @brief Get current position vector.
 * @return Position (translation) vector.
 */
template <>
inline std::array<double, 3> Pose::position() const
{
    return p_;
}

template <>
inline cv::Vec3d Pose::position() const
{
    cv::Vec3d p;
    p(0) = p_[0];
    p(1) = p_[1];
    p(2) = p_[2];
    return p;
}

template <>
inline std::array<double, 4> Pose::orientation() const
{
    return q_;
}

template <>
inline cv::Vec3d Pose::orientation() const
{
    // OpenCV's rvec is Euler axis multiplied by the angle value
    cv::Vec3d r{0, 0, 0};
    const double angle = 2 * std::acos(q_[w]);
    const double alpha = angle / std::sqrt(1 - q_[w] * q_[w]);
    if (!std::isnan(alpha)) {
        r[0] = alpha * q_[x];
        r[1] = alpha * q_[y];
        r[2] = alpha * q_[z];
    }

    return r;
}

// template <>
// inline std::array<double, 3> Pose::orientation() const
//{
//    // OpenCV's rvec is Euler axis multiplied by the angle
//    value
//    std::array<double, 3> r{{0, 0, 0}};
//    const double angle = 2 * std::acos(q_[w]);
//    const double alpha = angle / std::sqrt(1 - q_[w] * q_[w]);
//    if (!std::isnan(alpha)) {
//        r[0] = alpha * q_[x];
//        r[1] = alpha * q_[y];
//        r[2] = alpha * q_[z];
//    }
//
//    return r;
//}

template <>
inline cv::Matx33d Pose::orientation() const
{
    cv::Matx33d R;
    Q2R // Macro
    return R;
}

template <>
inline cv::Matx44d Pose::orientation() const
{
    auto R = cv::Matx44d::zeros();
    Q2R // Macro
    R(3, 3)= 1;
    return R;
}

#ifdef EIGEN3_FOUND
template <>
inline Eigen::Quaterniond Pose::orientation() const
{
    return eig_orient_;
}

template <>
inline Eigen::Matrix3d Pose::orientation() const
{
    return eig_orient_.toRotationMatrix();
}
#endif

//** HELPERS **//

inline std::array<double, 3> Pose::toTaitBryan(const bool deg) const
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

/**
 * @brief This routine maps three values (x[0], x[1], x[2]) in the range [0,1]
 * into a 3x3 rotation matrix, M.  Uniformly distributed random variables x0,
 * x1, and x2 create uniformly distributed random rotation matrices.  To create
 * small uniformly distributed "perturbations", supply samples in the following
 * ranges:
 *     x[0] in [ 0, d ]
 *     x[1] in [ 0, 1 ]
 *     x[2] in [ 0, d ]
 * where 0 < d < 1 controls the size of the perturbation.  Any of the
 * random variables may be stratified (or "jittered") for a slightly more
 * even distribution.
 * @param x Perturbation vector.
 * @return Uniformly random rotiation matrix.
 * @author Jim Arvo, 1991. Modified by J. Newman 2017.
 */
inline cv::Matx33d randRotation(std::array<double, 3> x)
{
    float theta = x[0] * M_PI * 2; // Rotation about the pole (Z).
    float phi = x[1] * M_PI * 2;   // For direction of pole deflection.
    float z = x[2] * 2.0;          // For magnitude of pole deflection.

    // Compute a vector V used for distributing points over the
    // sphere
    // via the reflection I - V Transpose(V).  This formulation
    // of V
    // will guarantee that if x[1] and x[2] are uniformly
    // distributed,
    // the reflected points will be uniform on the sphere.  Note
    // that V
    // has length sqrt(2) to eliminate the 2 in the Householder
    // matrix.

    float r = std::sqrt(z);
    float Vx = std::sin(phi) * r;
    float Vy = std::cos(phi) * r;
    float Vz = std::sqrt(2.0 - z);

    // Compute the row vector S = Transpose(V) * R, where R is a
    // simple
    // rotation by theta about the z-axis.  No need to compute
    // Sz since
    // it's just Vz.

    float st = std::sin(theta);
    float ct = std::cos(theta);
    float Sx = Vx * ct - Vy * st;
    float Sy = Vx * st + Vy * ct;

    // Construct the rotation matrix  ( V Transpose(V) - I ) R,
    // which
    // is equivalent to V S - R.

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

/**
 * @brief Serialize pose.
 * @param p Position to serialize.
 * @param w JSON writer to serialize with.
 * @param verbose Enable verbose serialization. Defaults to true.
 * @note Precision template parameter defaults to 3 indicating 0.001 pixels
 * or 1 mm precision
 */
template <typename Writer, size_t Precision>
void serializePose(const Pose &p, Writer &writer, bool verbose)
{
    writer.SetMaxDecimalPlaces(Precision);

    writer.StartObject();

    // Sample number
    writer.String("tick");
    writer.Uint64(p.sample_.count());

    writer.String("usec");
    writer.Uint64(p.sample_.microseconds().count());

    // Coordinate system
    writer.String("unit");
    writer.Int(static_cast<int>(p.unit_of_length));

    // Has the object been found?
    writer.String("found");
    writer.Bool(p.found);

    if (p.found || verbose) {
        writer.String("position");
        writer.StartArray();
        writer.Double(p.p_[0]);
        writer.Double(p.p_[1]);
        writer.Double(p.p_[2]);
        writer.EndArray(3);
    }

    if (p.found || verbose) {
        writer.String("orientation");
        writer.StartArray();
        writer.Double(p.q_[w]);
        writer.Double(p.q_[x]);
        writer.Double(p.q_[y]);
        writer.Double(p.q_[z]);
        writer.EndArray(3);
    }

    // Region
    writer.String("reg_ok");
    writer.Bool(p.region_valid);

    if (p.region_valid || verbose) {
        writer.String("reg");
        writer.String(p.region);
    }

    writer.EndObject();
}

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
       << "Position (" << unit << "): ["
       << p.p_[0] << " " << p.p_[1] << " " << p.p_[2] << "]\n"
       << "Orientation: ["
       << p.q_[w] << " " << p.q_[x] << " " << p.q_[y] << " " << p.q_[z] << "]";

    return os;
}

} /* namespace oat */
#endif /* OAT_POSE_H */
