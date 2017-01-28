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
static constexpr size_t POSE_NPY_DTYPE_BYTES {88};
static constexpr char POSE_NPY_DTYPE[]{"[('tick', '<u8'),"
                                       "('usec', '<u8'),"
                                       "('unit', '<i4'),"
                                       "('found', '<i1'),"
                                       "('position', 'f8', (3)),"
                                       "('orientation', 'f8', (4)),"
                                       "('in_region', '<i1'),"
                                       "('region', 'a10')]"};

// Forward decl.
class Pose;

/** 
 * @brief 
 * @param p
 * @param writer
 * @param verbose
 */
template <typename Writer, size_t Precision = 3>
void serializePose(const Pose &p, Writer &writer, bool verbose = true);

/** 
 * @brief 
 * @param os
 * @param p
 * @return 
 */
std::ostream &operator<<(std::ostream &os, const Pose &p);

/** 
 * @brief 
 * @param 
 * @return 
 */
cv::Matx33d randRotation(std::array<double, 3> x);

/**
 * @brief Pack a pose object into a byte array.
 * @param Position2D Position to pack into a byte array.
 * @return Byte arrayByte array.
 */
std::vector<char> packPose(const Pose &p);

/**
 * @brief Object pose.
 */
class Pose {


    template <typename Writer, size_t Precision>
    friend void serializePose(const Pose &p, Writer &w, bool verbose);
    friend std::vector<char> packPose(const Pose &);
    friend std::ostream &operator<<(std::ostream &os, const Pose &p);

public:
    enum class DistanceUnit {
        Pixels = 0, //!< Position measured in pixels. Origin is
                    //! upper left.
        Meters,     //!< Position measured in units specified via
                    //! homography
    };

    static constexpr size_t REGION_LEN {10};

    Pose() = default;

    Pose(const Pose &p)
    : unit_of_length(p.unit_of_length)
    , region_valid(p.region_valid)
    , found(p.found)
    , sample_(p.sample_)
    , q_(p.q_)
    , p_(p.p_)
    {
        std::strncpy(region, p.region, REGION_LEN);
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
        std::strncpy(region, rhs.region, REGION_LEN);
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
    bool region_valid {false};
    char region[REGION_LEN] {0}; //!< Categorical position label (e.g. "North West")

    // Sample information
    void set_sample(const Sample &val) { sample_ = val; }
    void set_rate_hz(const double rate_hz) { sample_.set_rate_hz(rate_hz); }
    double sample_period_sec() const { return sample_.period_sec().count(); }
    uint64_t sample_count(void) const { return sample_.count(); }
    uint64_t sample_usec(void) const { return sample_.microseconds().count(); }
    void incrementSampleCount() { sample_.incrementCount(); }
    //void incrementSampleCount(USec us) { sample_.incrementCount(us); }

    // Was pose estimation successful?
    bool found{false};

    // To understand type signatures of these function
    // templates:
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
    { (void)p; static_assert(sizeof(T) == 0, "Invalid call to set_position()"); }

    template <typename T>
    T position() const
    { static_assert(sizeof(T) == 0, "Invalid call to position()"); }

    // Get axis angle representation. Cannot use a template specialization
    // straightforwardly because this has the same return type as rvec.
    std::array<double, 3> toTaitBryan(const bool deg = false) const;

protected:
    /**
     * @brief Current time sample.
     */
    oat::Sample sample_;

    /**
     * @brief Orientation, represented as a Qauarterion, [x, y, z, w(real)].
     * Initialized to identity.  
     * @note Memory layout maps that of Eigen::Quaterniond
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
    // Probably some way to go directly to quaternion from rvec, but who cares
    // I guess
    cv::Matx33d R;
    cv::Rodrigues(r, R);
    R2Q // macro
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

}      /* namespace oat */
#endif /* OAT_POSE_H */
