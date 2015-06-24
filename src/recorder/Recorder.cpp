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

#include <chrono>
#include <ctime>
#include <iomanip>
#include <mutex>

#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <boost/dynamic_bitset.hpp>
#include <vector>

#include "Recorder.h"

namespace bfs = boost::filesystem;

Recorder::Recorder(const std::vector<std::string>& position_source_names,
        const std::vector<std::string>& frame_source_names,
        std::string save_path,
        std::string file_name,
        const bool& append_date,
        const int& frames_per_second,
        bool overwrite) :
  save_path(save_path)
, file_name(file_name)
, append_date(append_date)
, allow_overwrite(overwrite)
, running(true)
, frames_per_second(frames_per_second)
, number_of_frame_sources(frame_source_names.size())
, frame_read_required(number_of_frame_sources)
, number_of_position_sources(position_source_names.size())
, position_read_required(number_of_position_sources)
, position_labels(position_source_names)
, sources_eof(false) {

    // First check that the save_path is valid
    bfs::path path(save_path.c_str());
    if (!bfs::exists(path) || !bfs::is_directory(path)) {
        std::cout << "Warning: requested recording path, " + save_path + ", "
                << "does not exist, or is not a valid directory.\n"
                << "attempting to use the current directory instead.\n";
        save_path = bfs::current_path().c_str();
    }
    
    // Start recorder name construction
    name = "recorder[" ;
    
    // Create recording timestamp
    std::time_t raw_time;
    struct tm * time_info;
    char buffer[100];
    std::time(&raw_time);
    time_info = std::localtime(&raw_time);
    std::strftime(buffer, 80, "%F-%H-%M-%S", time_info);
    std::string date_now = std::string(buffer);

    // Setup position sources
    if (!position_source_names.empty()) {

        name += position_source_names[0];
        if (position_source_names.size() > 1)
            name += "...";
        
        for (auto &s : position_source_names) {

            position_sources.push_back(new oat::SMClient<oat::Position2D>(s));
            source_positions.push_back(new oat::Position2D);
        }

        // Create a single position file
        std::string posi_fid;
        if (append_date)
            posi_fid = file_name.empty() ?
            (save_path + "/" + date_now + "_" + position_source_names[0]) :
            (save_path + "/" + date_now + "_" + file_name);
        else
            posi_fid = file_name.empty() ?
            (save_path + "/" + position_source_names[0]) :
            (save_path + "/" + file_name);

        posi_fid = posi_fid + ".json";

        if (!allow_overwrite) {
            checkFile(posi_fid);
        }
        
        position_fp = fopen(posi_fid.c_str(), "wb");
        if (!position_fp) {
            std::cerr << "Error: unable to open, " + posi_fid + ". Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }

        file_stream = new rapidjson::FileWriteStream(position_fp, position_write_buffer, sizeof (position_write_buffer));
        json_writer.Reset(*file_stream);
        
        // Main object, end this object before write flush
        json_writer.StartObject();
        
        // Coordinate system
        json_writer.String("oat_version");
        json_writer.String("1.0");
        
        // Complete header object
        json_writer.String("header");
        writePositionFileHeader(date_now, frames_per_second, position_source_names);
        
        // Start data object
        json_writer.String("positions");
        json_writer.StartArray();
    }

    // Create a video writer, file, and buffer for each image stream
    uint32_t idx = 0;
    if (!frame_source_names.empty()) {
        
        name += frame_source_names[0];
        if (frame_source_names.size() > 1)
            name += "...";

        for (auto &frame_source_name : frame_source_names) {

            // Generate file name for this video
            std::string frame_fid;
            if (append_date)
                frame_fid = file_name.empty() ?
                (save_path + "/" + date_now + "_" + frame_source_name) :
                (save_path + "/" + date_now + "_" + file_name + "_" + frame_source_name);
            else
                frame_fid = file_name.empty() ?
                (save_path + "/" + frame_source_name) :
                (save_path + "/" + file_name);

            frame_fid = frame_fid + ".avi";

            if (!allow_overwrite) {
                checkFile(frame_fid);
            }

            video_file_names.push_back(frame_fid);
            frame_sources.push_back(new oat::MatClient(frame_source_name));
            
            frame_write_buffers.push_back(new
                    boost::lockfree::spsc_queue
                    < cv::Mat, boost::lockfree::capacity < frame_write_buffer_size> >);
            video_writers.push_back(new cv::VideoWriter());


            // Spawn frame writer threads and synchronize to incoming data
            frame_write_mutexes.push_back(new std::mutex());
            frame_write_condition_variables.push_back(new std::condition_variable());
            frame_write_threads.push_back(new std::thread(&Recorder::writeFramesToFileFromBuffer, this, idx++));

        }    
    } 
    
    frame_read_required.set();
    position_read_required.set();
    
    name +="]";
}

Recorder::~Recorder() {

    // Set running to false to trigger thread join
    running = false;
    for (auto &value : frame_write_condition_variables) {
        value->notify_one();
    }
    
    // Join all threads
    // Free all dynamically allocated resources
    for (auto &value : frame_write_threads) {
        value->join();
        delete value;
    }
    
    for (auto &value : video_writers) {
        delete value;
    }

    for (auto &value : frame_write_mutexes) {
        delete value;
    }

    for (auto &value : frame_write_condition_variables) {
        delete value;
    }

    for (auto &value : frame_write_buffers) {
        delete value;
    }

    for (auto &value : position_sources) {
        delete value;
    }
    
    for (auto &value : frame_sources) {
        delete value;
    }

    if (position_fp) {
        json_writer.EndArray();
        json_writer.EndObject();
        file_stream->Flush();
        delete file_stream;
    }
}

bool Recorder::writeStreams() {

    // Make sure all sources are still running
    for (int i = 0; i < number_of_frame_sources; i++) {

        sources_eof |= (frame_sources[i]->getSourceRunState()
                == oat::ServerRunState::END);
    }
    
    for (int i = 0; i < number_of_position_sources; i++) {

        sources_eof |= (position_sources[i]->getSourceRunState()
                == oat::ServerRunState::END);
    }
    
    boost::dynamic_bitset<>::size_type i = frame_read_required.find_first();
    
    while( i < number_of_frame_sources) {

        // Check if we need to read frame_client_idx, or if the read has been
        // performed already
        frame_read_required[i] = !frame_sources[i]->getSharedMat(current_frame);

        if (!frame_read_required[i]) {
            
            // Push newest frame into client N's queue
            frame_write_buffers[i]->push(current_frame);

            // Notify a writer thread that there is new data in the queue
            frame_write_condition_variables[i]->notify_one();
        }
        
        i = frame_read_required.find_next(i);
    }
    
    // Position source iterator
    i = position_read_required.find_first();

    // Get current positions
    while( i < number_of_position_sources) {
        
        // Check if we need to read position_client_idx, or if the read has been
        // performed already
        position_read_required[i] = 
                !position_sources[i]->getSharedObject(*source_positions[i]);
        
        i = position_read_required.find_next(i);
    }
    
    // If we have not finished reading _any_ of the clients, we cannot proceed
    if (frame_read_required.none() && position_read_required.none()) {
    
        // Reset the frame and position client read counter
        position_read_required.set();
        frame_read_required.set();

        // Write the frames to file
        writePositionsToFile();
    } 
    
    return sources_eof;
}

void Recorder::writeFramesToFileFromBuffer(uint32_t writer_idx) {

    while (running) {

        std::unique_lock<std::mutex> lk(*frame_write_mutexes[writer_idx]);
        frame_write_condition_variables[writer_idx]->wait_for(lk, std::chrono::milliseconds(10));

        cv::Mat m;
        while (frame_write_buffers[writer_idx]->pop(m)) {

            if (!video_writers[writer_idx]->isOpened()) {
                initializeWriter(*video_writers[writer_idx],
                        video_file_names.at(writer_idx),
                        m);
            }

            video_writers[writer_idx]->write(m);
        }
    }
}

void Recorder::writePositionsToFile() {

    if (position_fp) {

        //json_writer.String("positions");
        json_writer.StartArray();

        int idx = 0;
        for (auto pos : source_positions) {

            json_writer.Uint(position_sources[idx]->get_current_time_stamp());
            pos->Serialize(json_writer, position_labels[idx]);
            ++idx;
        }

        json_writer.EndArray();
    }
}

void Recorder::writePositionFileHeader(const std::string& date, 
        const double sample_rate, 
        const std::vector<std::string>& sources) {
    
    json_writer.StartObject();
    
    json_writer.String("date");
    json_writer.String(date.c_str());
    
    json_writer.String("sample_rate_hz");
    json_writer.Double(sample_rate);
    
    json_writer.String("position_sources");
    json_writer.StartArray();
    for (auto &s : sources) {
        json_writer.String(s.c_str());
    }
    json_writer.EndArray();
    
    json_writer.EndObject();

}

void Recorder::initializeWriter(cv::VideoWriter& writer,
        const std::string& file_name,
        const cv::Mat& image) {

    // Initialize writer using the first frame taken from server
    int fourcc = CV_FOURCC('H', '2', '6', '4');
    writer.open(file_name, fourcc, frames_per_second, image.size());

}

bool Recorder::checkFile(std::string& file) {

    int i = 0;
    std::string original_file = file;
    bool file_exists = false;

    while (bfs::exists(file.c_str())) {

        ++i;
        bfs::path path(original_file.c_str());
        bfs::path parent_path = path.parent_path();
        bfs::path stem = path.stem();
        bfs::path extension = path.extension();

        std::string append = "_" + std::to_string(i);
        stem += append.c_str();

        // Recreate file name
        file = std::string(parent_path.generic_string()) +
                "/" +
                std::string(stem.generic_string()) +
                std::string(extension.generic_string());

    }

    if (i != 0) {
        std::cout << "Warning: " + original_file + " exists.\n"
                << "File renamed to: " + file + ".\n";
        file_exists = true;
    }

    return file_exists;
}
