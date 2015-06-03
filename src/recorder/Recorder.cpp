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

#include <string>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sys/stat.h>
#include <boost/filesystem.hpp>

#include "Recorder.h"

namespace bfs = boost::filesystem;

Recorder::Recorder(const std::vector<std::string>& position_source_names,
        const std::vector<std::string>& frame_source_names,
        std::string save_path,
        std::string file_name,
        const bool& append_date,
        const int& frames_per_second) :
  save_path(save_path)
, file_name(file_name)
, append_date(append_date)
, frames_per_second(frames_per_second)
, frame_client_idx(0)
, position_client_idx(0)
, position_labels(position_source_names) {

    // First check that the save_path is valid
    bfs::path path(save_path.c_str());
    if (!bfs::exists(path) || !bfs::is_directory(path)) {
        std::cout << "Warning: requested recording path, " + save_path + ", "
                  << "does not exist, or is not a valid directory.\n"
                  << "attempting to use the current directory instead.\n";
        save_path = bfs::current_path().c_str();
    }

    std::time_t raw_time;
    struct tm * time_info;
    char buffer[100];

    std::time(&raw_time);
    time_info = std::localtime(&raw_time);
    std::strftime(buffer, 80, "%F-%H-%M-%S", time_info);
    std::string date_now = std::string(buffer);

    // Setup position sources
    if (!position_source_names.empty()) {

        for (auto &name : position_source_names) {

            position_sources.push_back(new shmem::SMClient<datatypes::Position2D>(name));
            source_positions.push_back(new datatypes::Position2D);
        }

        // Create a single position file
        std::string posi_fid;
        if (append_date)
            posi_fid = save_path + "/" + date_now + "_" + file_name;
        else
            posi_fid = save_path + "/" + position_source_names[0];

        posi_fid = posi_fid + ".json";

        checkFile(posi_fid);

        position_fp = fopen(posi_fid.c_str(), "wb");
        if (!position_fp) {
            std::cerr << "Error: unable to open, " + posi_fid + ". Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
        
        file_stream = new rapidjson::FileWriteStream(position_fp, position_write_buffer, sizeof (position_write_buffer));
        json_writer.Reset(*file_stream);
        json_writer.StartArray();
    }

    // Create a video writer and file for each image stream
    for (auto &frame_source_name : frame_source_names) {

        // Generate file name for this video
        std::string frame_fid;
        if (append_date)
            frame_fid = save_path + "/" + date_now + "_" + file_name + "_" + frame_source_name;
        else
            frame_fid = save_path + "/" + file_name + "_" + frame_source_name;

        frame_fid = frame_fid + ".avi";

        checkFile(frame_fid);

        video_file_names.push_back(frame_fid);
        frame_sources.push_back(new shmem::MatClient(frame_source_name));
        frames.push_back(new cv::Mat);
        video_writers.push_back(new cv::VideoWriter());
    }
}

Recorder::~Recorder() {

    // Release all resources
    for (auto &frame_source : frame_sources) {
        delete frame_source;
    }

    for (auto &frame : frames) {
        delete frame;
    }

    for (auto &writer : video_writers) {
        writer->release();
        delete writer;
    }

    for (auto &position_source : position_sources) {
        delete position_source;
    }
    if (position_fp) {
        json_writer.EndArray();
        file_stream->Flush();
        delete file_stream;
    }
}

void Recorder::writeStreams() {

    // Get current frames
    while (frame_client_idx < frame_sources.size()) {

        if (!(frame_sources[frame_client_idx]->getSharedMat(*frames[frame_client_idx]))) {
            return;
        }

        frame_client_idx++;
    }

    // Get current positions
    while (position_client_idx < position_sources.size()) {

        if (!(position_sources[position_client_idx]->getSharedObject(*source_positions[position_client_idx]))) {
            return;
        }

        position_client_idx++;
    }

    // Reset the position client read counter
    frame_client_idx = 0;
    position_client_idx = 0;

    // Write the frames to file
    writeFramesToFile();
    writePositionsToFile();

}

void Recorder::writeFramesToFile() {

    // Cycle through video writers, write each image source to the corresponding
    // file
    int idx = 0;
    for (auto &writer : video_writers) {

        if (writer->isOpened()) {
            writer->write(*frames[idx]);
        } else {
            initializeWriter(*writer, video_file_names.at(idx), *frames[idx]);
            writer->write(*frames[idx]);
        }

        ++idx;
    }
}

void Recorder::writePositionsToFile() {

    json_writer.StartObject();

    json_writer.String("sample");
    json_writer.Uint(position_sources[0]->get_current_time_stamp());
    
    json_writer.String("positions");
    
    json_writer.StartArray();

    int idx = 0;
    for (auto pos : source_positions) {

        pos->Serialize(json_writer, position_labels[idx]);
        ++idx;
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
