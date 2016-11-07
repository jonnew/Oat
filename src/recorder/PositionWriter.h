//******************************************************************************
//* File:   PositionWriter.h
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

#ifndef OAT_POSITIONWRITER_H
#define OAT_POSITIONWRITER_H

#include "Writer.h"

#include <boost/lockfree/spsc_queue.hpp>
#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/shmemdf/Source.h"
#include "../../lib/utility/FileFormat.h"

namespace oat {
namespace blf = boost::lockfree;

/**
 * Position stream file writer.
 */
class PositionWriter : public Writer{
public:

    PositionWriter(const std::string &addr);

    ~PositionWriter();

    void configure(const oat::config::OptionTable &t,
                   const po::variables_map &vm) override;
    void touch() override { source_.touch(addr_); }
    void connect() override { source_.connect(); }
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
    using SPSCBuffer = boost::lockfree::spsc_queue<oat::Position2D,
                                                   blf::capacity<BUFFER_SIZE>>;
    /**
     * @brief Determines if indeterminate position data fields should be
     * written in spite of being indeterminate for sample parsing ease? e.g.
     * Should we write pos_xy when pos_ok = false?
     */
    bool concise_file_ {false};

    std::string path_ {""};
    SPSCBuffer buffer_;

    //// Timestamp clock
    //std::chrono::system_clock clock_;
    //std::chrono::system_clock::time_point start_;

    // Position file
    FILE * fd_ {nullptr};
    int64_t completed_writes_ {0};

    // JSON-specific
    void initializeJSON(const std::string &path);
    char position_write_buffer[65536];
    std::unique_ptr<rapidjson::FileWriteStream> file_stream_;
    rapidjson::PrettyWriter<rapidjson::FileWriteStream> json_writer_ {*file_stream_};

    // Binary-specific
    void initializeBinary(const std::string &path);
    bool use_binary_ {false};
    static constexpr int header_prefix_size_ {10};
    static constexpr int shape_end_byte_ {10};

    oat::Source<Position2D> source_;
};

}      /* namespace oat */
#endif /* OAT_POSITIONWRITER_H */
