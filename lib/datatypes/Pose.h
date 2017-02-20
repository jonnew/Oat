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

#include "Sample.h"

#include <vector>
#include <cstring>

#include <cmath>
#include <opencv2/calib3d.hpp>
#include <rapidjson/prettywriter.h>
#ifdef EIGEN3_FOUND
#include <Eigen/Core>
#include <Eigen/Geometry>
#endif

/**
 * @def q2R
 * @cite Animating rotation with quaternion curves, Ken Shoemake, SIGGRAPH '85
 * Proceedings of the 12th annual conference on Computer graphics and
 * interactive techniques Pages 245-25
 */
#define q2R                                                                    \
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
 * @def R2q
 * @cite Animating rotation with quaternion curves, Ken Shoemake, SIGGRAPH '85
 * Proceedings of the 12th annual conference on Computer graphics and
 * interactive techniques Pages 245-25
 */
#define R2q                                                                    \
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

namespace oat {

// Constants
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
 * @brief Serialize pose using specified Writer type. Used for JSON streaming.
 * @param p Pose to Serialize.
 * @param writer Writer to use for serialize.
 * @param verbose Serialize pose data fields even when they are not defined.
 * e.g. when the object is not found (found = false) or not in a region
 * (in_region = false). Useful for alignment and ease of parsing.
 * @note Precision template parameter defaults to 3 indicating 0.001 pixels
 * or 1 mm precision
 */
template <typename Writer, size_t Precision = 3>
void serializePose(const Pose &p, Writer &writer, bool verbose = true);

/**
 * @brief Serialze pose to standard stream.
 * @param os Stream.
 * @param p Pose to serailize.
 * @return Serialzed pose.
 */
std::ostream &operator<<(std::ostream &os, const Pose &p);

/**
 * @brief This routine maps three values (x[0], x[1], x[2]) in the range [0,1]
 * into a 3x3 rotation matrix, R.  Uniformly distributed random variables x0,
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
 * @return Uniformly random rotation matrix.
 * @author Jim Arvo, 1991. Modified by J. Newman 2017.
 */
cv::Matx33d randRotation(std::array<double, 3> x);

/**
 * @brief Pack a pose object into a byte array.
 * @param Position2D Position to pack into a byte array.
 * @return Byte arrayByte array.
 */
std::vector<char> packPose(const Pose &p);

/**
 * @brief Detected object pose.
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
        Meters,     //!< Position measured in meters.
    };

    enum class DOF {
        Zero = 0, //!< Zero degrees of freedom (fixed)
        One,      //!< One degrees of freedom (X)
        Two,      //!< Two degrees of freedom (XY)
        Three,    //!< Three degrees of freedom (XYZ)
    };

    static constexpr size_t region_max_char {10};

    Pose(){};

    Pose(const DistanceUnit u, const DOF p_dof, const DOF o_dof)
    : unit_of_length(u)
    , position_dof(p_dof)
    , orientation_dof(o_dof)
    {
        // Nothing
    }

    Pose(const Pose &p)
    : unit_of_length(p.unit_of_length)
    , position_dof(p.position_dof)
    , orientation_dof(p.orientation_dof)
    , in_region(p.in_region)
    , found(p.found)
    , sample_(p.sample_)
    , q_(p.q_)
    , p_(p.p_)
    {
        std::strncpy(region, p.region, region_max_char);
    }

    Pose(Pose &&p) = default;

    Pose &operator=(const Pose &rhs)
    {
        unit_of_length = rhs.unit_of_length;
        position_dof= rhs.position_dof;
        orientation_dof= rhs.orientation_dof;
        in_region = rhs.in_region;
        std::strncpy(region, rhs.region, region_max_char);
        found = rhs.found;
        sample_ = rhs.sample_;
        q_ = rhs.q_;
        p_ = rhs.p_;
        return *this;
    }

    Pose &operator=(Pose &&) = default;

    /**
     * @brief Produce a pose from thin air. This is used by pure SINKs (e.g.
     * posigen) that have no sample information to pass forward but instead are
     * responsible for creating it.
     * @param p Pose sample to set this one too. Associated timing info is
     * ignored.
     * @param usec Usec since production has started. Used to increment
     * associated timing info. If set to zero or not supplied, defaults to just
     * updating sample count without a time.
     */
    void produce(const Pose &p, Sample::Microseconds usec = Sample::Microseconds{0}) {
        unit_of_length = p.unit_of_length;
        position_dof = p.position_dof;
        orientation_dof = p.orientation_dof;
        in_region = p.in_region;
        std::strncpy(region, p.region, region_max_char);
        found = p.found;
        q_ = p.q_;
        p_ = p.p_;
        if (usec == Sample::Microseconds{0})
            sample_.incrementCount();
        else
            sample_.incrementCount(usec);
    }

    /**
     * @brief Unit of length used to specify rotation and translation vectors.
     */
    DistanceUnit unit_of_length{DistanceUnit::Pixels};

    /**
     * @brief Degrees of freedom encoded by the position component of the pose.
     */
    DOF position_dof{DOF::Three};

    /**
     * @brief Degrees of freedom encoded by the orientation component of the pose.
     */
    DOF orientation_dof{DOF::Three};

    bool in_region {false};      //!< Does pose fall in valid region
    char region[region_max_char] {0}; //!< Categorical position label (e.g. "North")

    // Sample information
    void set_sample(const Sample &val) { sample_ = val; }
    void set_rate_hz(const double rate_hz) { sample_.set_rate_hz(rate_hz); }
    double sample_period_sec() const { return sample_.period_sec().count(); }
    uint64_t sample_count(void) const { return sample_.count(); }
    uint64_t sample_usec(void) const { return sample_.microseconds().count(); }
    void incrementSampleCount() { sample_.incrementCount(); }
    void incrementSampleCount(Sample::Microseconds us) { sample_.incrementCount(us); }
    void resample(const double resample_ratio) { sample_.resample(resample_ratio); }

    // Was pose estimation successful?
    bool found{false};

    // To understand the const ref type signatures of these function templates:
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

    /**
     * @brief Convert orientation to Tait Bryan (Euler) angles.
     * @param deg If true, result is specified in degrees instead of radians.
     * @return Tait Bryan Angles (Yaw, Pitch, Roll)
     * @note Cannot use a template specialization straightforwardly because has
     * the same return type as rvec.
     */
    std::array<double, 3> toTaitBryan(const bool deg = false) const;

    /**
     * @brief Set the current orientation using Tait Bryan (Euler) angles.
     * @param angles Tait Bryan angles to use to set the orientation (Yaw, Pitch,
     * Roll).
     * @note Cannot use a template specialization straightforwardly because has
     * the same input type as rvec.
     */
    void fromTaitBryan(const std::array<double, 3> &angles,
                       const bool deg = false);

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
    R2q // Macro
}

template <>
inline void Pose::set_orientation(const cv::Vec3d &r)
{
    // Probably some way to go directly to quaternion from rvec, but who cares
    // I guess
    cv::Matx33d R;
    cv::Rodrigues(r, R);
    R2q // macro
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
    q2R // Macro
    return R;
}

template <>
inline cv::Matx44d Pose::orientation() const
{
    auto R = cv::Matx44d::zeros();
    q2R // Macro
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
    writer.String("in_region");
    writer.Bool(p.in_region);

    if (p.in_region || verbose) {
        writer.String("region");
        writer.String(p.region);
    }

    writer.EndObject();
}

}      /* namespace oat */
#endif /* OAT_POSE_H */
