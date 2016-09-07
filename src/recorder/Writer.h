//******************************************************************************
//* File:   Writer.h
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
//*****************************************************************************

#ifndef OAT_WRITER_H
#define OAT_WRITER_H

#include <string>

#include <boost/lockfree/spsc_queue.hpp>
#include <opencv2/videoio.hpp>
#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>

#include "../../lib/datatypes/Frame.h"
#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/FileFormat.h"

namespace oat {
namespace blf = boost::lockfree;

static constexpr int SAMPLE_BUFFER_SIZE {1000};

/**
 * Generic, abstract file writer for a single data source
 */
template <typename T>
class Writer {

using SPSCBuffer =
    boost::lockfree::spsc_queue<T, blf::capacity<SAMPLE_BUFFER_SIZE>>;

public:

    Writer(const std::string &path) :
      path_(path) 
    {
        if (!oat::checkWritePermission(path_))
            throw (std::runtime_error("Write permission denied for " + path_));
    }

    /**
     * @brief Create and initialize recording file(s). Must be called
     * before writeStreams.
     */
    virtual void initialize(const std::string &source_name,
                            const T &sample_template) = 0;

    /**
     * @brief Flush internal sample buffer to file.
     */
    virtual void write(void) = 0;

    /**
     * @brief Push a sample onto the internal, lock-free, thread-safe buffer
     * @return False if there is an overflow condition. True otherwise.
     */
    void push(const T &sample) {
        if (!buffer_.push(sample)) {
            throw (std::runtime_error("Record buffer overrun. You can:\n"
                                      " - decrease the sample rate\n"
                                      " - use multiple recorders on multiple disks\n"
                                      " - or, get a faster hard disk"));
        }
    }

protected:

    /** 
     * @brief Fully qualified to file this writer is writting to.
     */
    std::string path_ {""};

    /** 
     * @brief Lock-free, thread-safe buffer which is flushed to file with each call to write. 
     */
    SPSCBuffer buffer_;
};

}      /* namespace oat */
#endif /* OAT_WRITER_H */
