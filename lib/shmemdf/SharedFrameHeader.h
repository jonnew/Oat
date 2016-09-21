//******************************************************************************
//* File:   SharedFrameHeader.h
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

#ifndef OAT_SHAREDFRAMEHEADER_H
#define	OAT_SHAREDFRAMEHEADER_H

#include <atomic>
#include <boost/interprocess/managed_shared_memory.hpp>

#include "../datatypes/Color.h"

namespace oat {
namespace bip = boost::interprocess;

struct FrameParams {
    size_t cols  {0};
    size_t rows  {0};
    int type  {0};
    oat::PixelColor color {oat::PIX_BGR};
    size_t bytes {0};
};

/** Header to facilitate zero-copy oat::Frame exchange through shared
  * memory.
  *
  * This class contains everything required to pass Frames through shared
  * memory without a copy. Basically, this class contains two shmem handles:
  * data_ and sample_. These handles provide cross-process pointer access to
  * two blocks of shared memory, one for matrix data and other for sample count
  * and rate information. Non-pointer members allow construction of Frames at
  * source and sink end contain this data and sample information.
  */

class SharedFrameHeader {

    using handle_t = bip::managed_shared_memory::handle_t;

public :

    handle_t sample() const { return sample_; }
    handle_t data() const { return data_; }
    FrameParams params() const { return params_; }

    /**
     * Set header data fields.
     *
     * @param data Interprocess handle to matrix data pointer
     * @param sample Interprocess handle to frame sample struct pointer
     * @param rows Number of rows in the matrix
     * @param cols Number of columns in the matrix
     * @param type OpenCV cv::Mat type of the frame
     */
    void setParameters(const handle_t data,
                       const handle_t sample,
                       const size_t rows,
                       const size_t cols,
                       const int type,
                       const oat::PixelColor color)
    {
        data_ = data;
        sample_ = sample;
        params_.rows = rows;
        params_.cols = cols;
        params_.type = type;
        params_.color = color;
    }

private :

    // TODO: Should these be atomic? They should already be protected by
    // the semaphores wrapping critical sections in the code. I guess they
    // are manipulated by bind() and connect() methods without semaphore
    // protection though. But, only bind writes.

    // Matrix metadata
    FrameParams params_;

    // Interprocess matrix data and sample handles
    handle_t data_;
    handle_t sample_;
};

}       /* namespace oat */
#endif	/* OAT_SHAREDFRAMEHEADER_H */
