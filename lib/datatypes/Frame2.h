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

#include "Pixel.h"
#include "Token.h"

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <opencv2/core/mat.hpp>
//#ifdef EIGEN3_FOUND
//#include <Eigen/Core>
//#endif

#define OAT_DEFAULT_FPS 1000000

namespace oat {

/**
 * @brief Video frame.
 */
template <typename DataContainer,
          typename Allocator = std::allocator<Pixel::ValueT>>
class FrameBase : public Token {

public:
    FrameBase(const double ts_sec,
              const size_t cols,
              const size_t rows,
              const Pixel::Color color,
              const Allocator &alloc = Allocator())
    : Token(ts_sec)
    , storage(cols * rows * Pixel::depth(color), 0, alloc)
    , rows_(rows)
    , cols_(cols)
    , color_(color)
    {
        // Nothing
    }

    FrameBase(const Seconds ts,
              const size_t cols,
              const size_t rows,
              const Pixel::Color color,
              const Allocator &alloc = Allocator())
    : Token(ts)
    , storage(cols * rows * Pixel::depth(color), 0, alloc)
    , rows_(rows)
    , cols_(cols)
    , color_(color)
    {
        // Nothing
    }

    FrameBase(const FrameBase &f) = default;
    FrameBase &operator=(const FrameBase &f) = default;

    // TODO: Moves do not work through shmem because allocator is stateful.
    // get_allocator assumes stateless allocator, I think. This is why boost ipc
    // has its own vector implementation
    FrameBase(FrameBase &&f) //= default;
    : Token(std::move(f.period<Token::Seconds>()))
    , storage(std::move(f.storage)) //f.storage.get_allocator())
    , rows_(std::move(f.rows_))
    , cols_(std::move(f.cols_))
    , color_(std::move(f.color_))
    {
        // Nothing
        std::cout << "Use frame move ctor." << std::endl;
    }
    
    FrameBase &operator=(FrameBase &&rhs) = default;
    //{
    //    Token::operator=(rhs);
    //    cols_ = std::move(rhs.cols_);
    //    rows_ = std::move(f.rows_);
    //    //storage = std::move(f.storage), f.storage.get_allocator())
    //    return *this;
    //}

    // Data getter/setters
    // TODO: Eigen::Matrix return type overloads

    cv::Mat to() const
    {
        return cv::Mat(
            rows_, cols_, Pixel::cvType(color_), (void *)storage.data());
    }

    cv::Mat clone() const
    {
        return cv::Mat(rows_, cols_, Pixel::cvType(color_), (void *)storage.data())
            .clone();
    }

    void copyFrom(const cv::Mat &m)
    {
        // Check size
        assert(m.rows == static_cast<int>(rows_) && m.cols == static_cast<int>(cols_)
               && "cv::Mat and frame must be the same size.");

        // TODO: Check pixel type
        assert(m.elemSize() == Pixel::bytes(color_)
               && "cv::Mat and frame must have same element size.");

        memcpy(storage.data(), m.data, bytes());
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

    /**
     * @brief Potentially shmem-allocated vector containing frame data.
     * @warning User can mess with underlying data and make it conflict with
     * the cols-, rows-, depth-determined storage requirements of the Frame.
     */
    DataContainer storage;

private:
    // Frame size
    const size_t rows_{0};
    const size_t cols_{0};

    // Pixel element color
    const Pixel::Color color_;


//#ifdef EIGEN3_FOUND
//    Eigen::Map<Matrix4f, RowMajor, Stride<3, 1>> red(cvT.data);
//    Eigen::Map<Matrix4f, RowMajor, Stride<3, 1>> green(cvT.data + 1);
//    Eigen::Map<Matrix4f, RowMajor, Stride<3, 1>> blue(cvT.data + 2);
//#endif
};

using StorageT = std::vector<Pixel::ValueT>;
using Frame = FrameBase<StorageT>;

namespace bip = boost::interprocess;
using SharedFrameAllocator
    = bip::allocator<oat::Pixel::ValueT,
                     bip::managed_shared_memory::segment_manager>;

using SharedStorageT = bip::vector<Pixel::ValueT, SharedFrameAllocator>;
using SharedFrame = FrameBase<SharedStorageT, SharedFrameAllocator>;

// TODO: This has an extra copy on the return since std::vectors are copy by
// value
namespace frame {

    inline Frame getLocal(const SharedFrame &from)
    {
        Frame to(from.period<Token::Seconds>(), from.cols(), from.rows(), from.color());
        memcpy((void *)to.storage.data(), (void *)from.storage.data(), to.bytes());
        return to;
    }
}      /* namespace frame */
}      /* namespace oat */
#endif /* OAT_FRAME_H */
