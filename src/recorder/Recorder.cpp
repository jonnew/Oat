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

#include "Recorder.h"

Recorder::Recorder(const std::vector<std::string>& position_source_names,
        const std::vector<std::string>& frame_source_names,
        const std::string& save_path,
        const bool& append_date,
        const int& frames_per_second) :
  save_path(save_path)
, append_date(append_date)
, frames_per_second(frames_per_second)
, frame_client_idx(0) {

    std::time_t raw_time;
    struct tm * time_info;
    char buffer[80];

    std::time(&raw_time);
    time_info = std::localtime(&raw_time);
    std::strftime(buffer, 80, "%F-%H-%M-%S_", time_info); 
    std::string date_now = std::string(buffer);

    // TODO: Create a single position file

    // Create a video writer and file for each image stream
    for (auto &frame_source_name : frame_source_names) {

        // Generate file name for this video
        std::string this_fid;
        if (append_date)
            this_fid = save_path + "/" + date_now + frame_source_name + ".avi";
        else
            this_fid = save_path + "/" + frame_source_name + ".avi";

        // TODO: if the file exists, append a numeral

        video_file_names.push_back(this_fid);
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

}

void Recorder::writeStreams() {

    // Get the current positions
    while (frame_client_idx < frame_sources.size()) {

        if (!(frame_sources[frame_client_idx]->getSharedMat(*frames[frame_client_idx]))) {
            return;
        }

        frame_client_idx++;
    }

    // Reset the position client read counter
    frame_client_idx = 0;

    // Write the frames to file
    writeFramesToFile();

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

        idx++;
    }
}

void Recorder::initializeWriter(cv::VideoWriter& writer,
        const std::string& file_name,
        const cv::Mat& image) {

    // Initialize writer using the first frame taken from server
    int fourcc = CV_FOURCC('H', '2', '6', '4');
    writer.open(file_name, fourcc, frames_per_second, image.size());

}
