//******************************************************************************
//* File:   FrameWriter.h
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

#ifndef OAT_FRAMEWRITER_H
#define OAT_FRAMEWRITER_H

#include "Writer.h"

#include <boost/lockfree/spsc_queue.hpp>
#include <opencv2/videoio.hpp>

#include "../../lib/datatypes/Frame.h"
#include "../../lib/utility/FileFormat.h"

namespace oat {
namespace blf = boost::lockfree;

class FrameWriter : public Writer {

public:
    using Writer::Writer;

    void configure(const oat::config::OptionTable &t,
                   const po::variables_map &vm) override;
    void touch() override { source_.touch(addr_); }
    oat::SourceState connect() override;
    double sample_period_sec() override
    {
        return source_.retrieve()->sample_period_sec();
    }
    oat::NodeState wait() override { return source_.wait(); }
    void post(void) override { source_.post(); }

    void initialize(const std::string &path) override;
    void write(void) override;
    void push(void) override;
    void deleteFile() override
    {
        if (!path_.empty())
            std::remove(path_.c_str());
    }

private:
    using SPSCBuffer
        = boost::lockfree::spsc_queue<oat::Frame, blf::capacity<BUFFER_SIZE>>;
    SPSCBuffer buffer_;

    // Video writer and required parameters
    std::string path_ {""};
    int fourcc_ {0}; // Default to uncompressed
    double fps_;
    oat::FrameParams frame_params_;
    cv::VideoWriter video_writer_;

    // The held frame source
    oat::Source<oat::Frame> source_;
};

}      /* namespace oat */
#endif /* OAT_FRAMEWRITER_H */
