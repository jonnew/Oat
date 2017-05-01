//******************************************************************************
//* File:   Frame.h
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

#ifndef OAT_FRAME_H
#define	OAT_FRAME_H

#include "OatConfig.h" // EIGEN3_FOUND

#include <algorithm>
#include <iterator>

#include "Pixel.h"
#include "Token.h"

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <opencv2/core/mat.hpp>
//#ifdef EIGEN3_FOUND
//#include <Eigen/Core>
//#endif

#define OAT_DEFAULT_FPS Token::Seconds(1000000)

namespace oat {

/**
 * @brief Video frame.
 */
template <typename DataContainer,
          typename Allocator = std::allocator<Pixel::ValueT>>
class FrameBase : public Token {

    // This class's frame type
    using FrameBaseT = FrameBase<DataContainer, Allocator>;

    // Friends with other versions of this class template's move and copy functions
    template <typename From, typename To>
    friend To moveFrame(From &from);

    template <typename From, typename To>
    friend To copyFrame(const From &from);

    friend bool compare(const FrameBaseT &f0, const FrameBaseT &f1);

public:
    FrameBase()
    {
        // Default
    }

    FrameBase(const Seconds ts,
              const size_t rows,
              const size_t cols,
              const Pixel::Color color,
              const Allocator &alloc = Allocator())
    : Token(ts)
    , storage(alloc)
    , rows_(rows)
    , cols_(cols)
    , color_(color)
    {
        // No initialization, no cost
        storage.reserve(cols * rows * Pixel::depth(color));
    }

    // Data getter/setters
    // TODO: Eigen::Matrix return type overloads

    cv::Mat mat() const
    {
        return cv::Mat(
            rows_, cols_, Pixel::cvType(color_), (void *)storage.data());
    }

    cv::Mat cloneMat() const
    {
        return cv::Mat(rows_, cols_, Pixel::cvType(color_), (void *)storage.data())
            .clone();
    }

    void copyFrom(const cv::Mat &mat)
    {
        // Check size
        assert(mat.rows == static_cast<int>(rows_) && mat.cols == static_cast<int>(cols_)
               && "cv::Mat and frame must be the same size.");

        assert(mat.elemSize() == Pixel::bytes(color_)
               && "cv::Mat and frame must have same element size.");

        // TODO: Check pixel type

        // Copy data, make sure that storage is correct size
        storage.resize(cols_ * rows_ * Pixel::depth(color_));
        memcpy((void *)storage.data(), (void *)mat.data, bytes());
    }

    /**
     * @brief Number of rows in storage matrix
     * @return rows
     */
    size_t cols() const { return cols_; }

    /**
     * @brief Number of columns in storage matrix
     * @return rows
     */
    size_t rows() const { return rows_; }

    /**
     * @brief Color of pixel elements
     * @return color
     */
    Pixel::Color color() const { return color_; }

    /**
     * @brief Bytes of data required to store frame data in underlying container.
     * @return Number of bytes required for data storage. Does not include
     * header information.
     */
    size_t bytes() const
    {
        return rows_ * cols_ * Pixel::bytes(color_);
    }


    static bool compare(const FrameBaseT &f0, const FrameBaseT &f1)
    {
        return (f0.cols_ == f1.cols_ 
                && f0.rows_ == f1.rows_
                && f0.color_ == f1.color_);
    }


    /**
     * @brief Potentially shmem-allocated vector containing frame data.
     * @warning This class can mess with underlying data and make it conflict
     * with the cols-, rows-, depth-determined storage requirements of the
     * Frame. 
     */
    DataContainer storage;

private:
    // Frame size
    size_t rows_{0};
    size_t cols_{0};

    // Pixel element color
    Pixel::Color color_;

//#ifdef EIGEN3_FOUND
//    Eigen::Map<Matrix4f, RowMajor, Stride<3, 1>> red(cvT.data);
//    Eigen::Map<Matrix4f, RowMajor, Stride<3, 1>> green(cvT.data + 1);
//    Eigen::Map<Matrix4f, RowMajor, Stride<3, 1>> blue(cvT.data + 2);
//#endif
};

// TODO: Ensure that the copy returned by these functions is eleted
template <typename From, typename To>
static To moveFrame(From &from)
{
    To to(from.template period<Token::Seconds>(),
          from.rows(),
          from.cols(),
          from.color());
    to.setTime(from.tick(), from.template time<Token::Seconds>());
    std::move(from.storage.begin(),
              from.storage.end(),
              std::back_inserter(to.storage));
    return to;
}

template <typename From, typename To>
static To copyFrame(const From &from)
{
    To to(from.template period<Token::Seconds>(),
          from.rows(),
          from.cols(),
          from.color());
    to.setTime(from.tick(), from.template time<Token::Seconds>());
    std::copy(from.storage.begin(),
              from.storage.end(),
              std::back_inserter(to.storage));
    return to;
}

// Frame - local use
using StorageT = std::vector<Pixel::ValueT>;
using Frame = FrameBase<StorageT>;

// SharedFrame - shmem use
namespace bip = boost::interprocess;
using SharedFrameAllocator
    = bip::allocator<oat::Pixel::ValueT,
                     bip::managed_shared_memory::segment_manager>;
using SharedStorageT = bip::vector<Pixel::ValueT, SharedFrameAllocator>;
using SharedFrame = FrameBase<SharedStorageT, SharedFrameAllocator>;

}      /* namespace oat */
#endif /* OAT_FRAME_H */
