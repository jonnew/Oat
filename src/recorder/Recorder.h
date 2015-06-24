//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
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

#ifndef RECORDER_H
#define RECORDER_H

#include <atomic>
#include <condition_variable>
#include <string>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/lockfree/spsc_queue.hpp>

#include "../../lib/rapidjson/filewritestream.h"
#include "../../lib/rapidjson/prettywriter.h"
#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/SMClient.h"
#include "../../lib/datatypes/Position2D.h"

class Recorder {
public: 

    // Both positions and images
    Recorder(const std::vector<std::string>& position_source_names,
            const std::vector<std::string>& frame_source_names,
            std::string save_path = ".",
            std::string file_name = "",
            const bool& append_date = false,
            const int& frames_per_second = 30,
            const bool overwrite = false);

    ~Recorder();

    // Recorder must be configurable
    void configure(const std::string& config_file, const std::string& config_key);

    // Recorder must be able to write all requested streams to file(s)
    bool writeStreams(void);
    
    std::string get_name(void) { return name; }

private:
    
    std::string name;

    // General file name formatting
    std::string save_path;
    std::string file_name;
    const bool append_date;
    const bool allow_overwrite;
    
    // File writer in running state (i.e. all threads should remain responsive for
    // new data coming down the pipeline)
    std::atomic<bool> running; 
    
    // Video files
    const int frames_per_second;
    std::vector<std::string> video_file_names;
    std::vector<cv::VideoWriter*> video_writers;
    
    // Position file
    FILE* position_fp;
    char position_write_buffer[65536];
    rapidjson::FileWriteStream * file_stream;
    rapidjson::Writer<rapidjson::FileWriteStream> json_writer;

    // Frame sources
    boost::dynamic_bitset<>::size_type number_of_frame_sources;
    std::vector<oat::MatClient* > frame_sources;
    cv::Mat current_frame;
    boost::dynamic_bitset<> frame_read_required;
    static const int frame_write_buffer_size = 100;

    std::vector< std::thread* > frame_write_threads;
    std::vector< std::mutex* > frame_write_mutexes;
    std::vector< std::condition_variable* > frame_write_condition_variables;
    std::vector< boost::lockfree::spsc_queue
               < cv::Mat, boost::lockfree::capacity
               < frame_write_buffer_size> > * > frame_write_buffers;
    
    // Position sources
    boost::dynamic_bitset<>::size_type number_of_position_sources;
    std::vector<oat::SMClient<oat::Position2D>* > position_sources;
    std::vector<oat::Position2D* > source_positions;
    boost::dynamic_bitset<> position_read_required;
    std::vector<std::string> position_labels;
    
    // Sources EOF flag
    bool sources_eof;

    void openFiles(const std::vector<std::string>& save_path,
            const bool& save_positions,
            const bool& save_images);

    void initializeWriter(cv::VideoWriter& writer,
            const std::string& file_name,
            const cv::Mat& image);

    bool checkFile(std::string& file);

    void writeFramesToFileFromBuffer(uint32_t writer_idx);
    void writePositionsToFile(void);
    void writePositionFileHeader(
        const std::string& date, 
        const double sample_rate, 
        const std::vector<std::string>& sources);

};

#endif // RECORDER_H
