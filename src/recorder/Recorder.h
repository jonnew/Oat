//******************************************************************************
//* File:   Recorder.h
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

#ifndef OAT_RECORDER_H
#define OAT_RECORDER_H

#include <atomic>
#include <condition_variable>
#include <string>
#include <thread>
#include <boost/dynamic_bitset.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>

#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/SharedCVMat.h"
#include "../../lib/datatypes/Position2D.h"

namespace oat {

static constexpr int FRAME_WRITE_BUFFER_SIZE {1000};
static constexpr int POSITION_WRITE_BUFFER_SIZE {65536};

/**
 * Position and frame recorder.
 */
class Recorder {
public:

    using PositionSource = std::pair <
                                std::string,
                                std::unique_ptr<oat::Source<oat::Position2D> >
                            >;

    using psvec_size_t = std::vector<PositionSource>::size_type;
    using pvec_size_t = std::vector<oat::Position2D>::size_type;

    using FrameSource = std::pair <
                            std::string,
                            std::unique_ptr<oat::Source<oat::SharedCVMat> >
                        >;

    using fvec_size_t = std::vector<FrameSource>::size_type;

    /**
     * Position and frame recorder.
     * @param position_source_addresses Addresses specifying position SOURCES to record
     * @param frame_source_addresses Addresses specifying frame SOURCES to record
     * @param save_path Directory to which files will be written
     * @param file_name Base file name
     * @param append_date Should the date be prepended to the file name
     * @param frames_per_second Frame rate if video is recorded
     * @param overwrite Should a file with the same name be overwritten
     */
    Recorder(const std::vector<std::string> &position_source_addresses,
             const std::vector<std::string> &frame_source_addresses,
             std::string save_path = ".",
             std::string file_name = "",
             const bool prepend_date = false,
             const int frames_per_second = 30,
             const bool overwrite = false);

    ~Recorder();

    /**
     * Recorder SOURCEs must be able to connect to a NODEs from
     * which to receive positions and frames.
     */
    void connectToNodes(void);

    /**
     * Collect frames and positions from SOURCES. Write frames and positions to file.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    bool writeStreams(void);

    /**
     * Get recorder name
     * @return name
     */
    std::string name(void) { return name_; }

private:

    // Name of this recorder
    std::string name_;

    // General file name formatting
    std::string save_path_;
    std::string file_name_;
    const bool append_date_;
    const bool allow_overwrite_;

    // File writer in running state (i.e. all threads should remain responsive for
    // new data coming down the pipeline)
    std::atomic<bool> running {true};

    // Video files
    const int frames_per_second_;
    std::vector< std::string > video_file_names_;
    std::vector< std::unique_ptr
               < cv::VideoWriter > > video_writers_;

    // Position file
    FILE * position_fp {nullptr};
    char position_write_buffer[POSITION_WRITE_BUFFER_SIZE];
    std::unique_ptr<rapidjson::FileWriteStream> file_stream_;
    rapidjson::PrettyWriter<rapidjson::FileWriteStream> json_writer_ {*file_stream_};

    // Frame sources
    std::vector<FrameSource> frame_sources_;
    boost::dynamic_bitset<> frame_read_required_;

    // Multi video writer multi-threading
    std::vector< std::unique_ptr
               < std::thread > > frame_write_threads_;
    std::vector< std::unique_ptr
               < std::mutex > > frame_write_mutexes_;
    std::vector< std::unique_ptr
               < std::condition_variable > > frame_write_condition_variables_;
    std::vector< std::unique_ptr
               < boost::lockfree::spsc_queue
               < cv::Mat, boost::lockfree::capacity
               < FRAME_WRITE_BUFFER_SIZE > > > > frame_write_buffers_;

    // Position sources
    std::vector<oat::Position2D> positions_;
    std::vector<uint64_t> position_write_number_;
    std::vector<PositionSource> position_sources_;

    //boost::dynamic_bitset<>::size_type number_of_position_sources_;
//    std::vector<PositionSource> position_sources_;
//    std::vector< std::unique_ptr
//               < oat::Position2D > > positions_;
//    boost::dynamic_bitset<> position_read_required_;

    // SOURCES EOF flag
    bool sources_eof {false};

    void openFiles(const std::vector<std::string>& save_path,
                   const bool& save_positions,
                   const bool& save_images);

    void initializeWriter(cv::VideoWriter& writer,
                          const std::string& file_name,
                          const cv::Mat& image);

    bool checkFile(std::string& file);

    void writeFramesToFileFromBuffer(uint32_t writer_idx);
    void writePositionsToFile(void);
    void writePositionFileHeader(const std::string& date,
                                 const double sample_rate,
                                 const std::vector<std::string>& sources);
};

}      /* namespace oat */
#endif /* OAT_RECORDER_H */
