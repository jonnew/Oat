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
#include <boost/program_options.hpp>
#include <opencv2/videoio.hpp>
#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>

#include "../../lib/shmemdf/Node.h"
#include "../../lib/shmemdf/Source.h"
#include "../../lib/utility/FileFormat.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {
namespace blf = boost::lockfree;
namespace po = boost::program_options;

class Writer {

public:
    /**
     * Abstract file writer for a single data source
     */
    Writer(const std::string &addr)
    : addr_(addr)
    {
        // Nothing
    }

    virtual ~Writer() { }

    // Program option handling
    virtual void configure(const oat::config::OptionTable &t,
                           const po::variables_map &vm) = 0;

    // Stuff for manipulating held source
    virtual void touch(void) = 0;
    virtual oat::SourceState connect(void) = 0;
    virtual oat::NodeState wait(void) = 0;
    virtual void post(void) = 0;
    virtual double sample_period_sec(void) = 0;

    /**
     * @brief Create and initialize recording file. Must be called
     * before writeStreams.
     */
    virtual void initialize(const std::string &path) = 0;

    /**
     * Push a new sample onto the write queue
     */
    virtual void push(void) = 0;

    /**
     * @brief Flush internal sample buffer to file.
     */
    virtual void write(void) = 0;

    /**
     * @brief Delete file
     */
    virtual void deleteFile(void) = 0;

    std::string addr(void) const { return addr_; }

protected:
    static constexpr int BUFFER_SIZE {1000};
    static const char OVERRUN_MSG[];

    /**
     * @breif Address of shmem for held source
     */
    std::string addr_;

    /**
     * @brief Allow file overwrite if true. If false append numerical index to
     * file name to make it unique.
     */
    bool allow_overwrite_ {false};
};

}      /* namespace oat */
#endif /* OAT_WRITER_H */
