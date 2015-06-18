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

#include "Viewer.h"

#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../lib/shmem/SyncSharedMemoryObject.h"
#include "../../lib/shmem/MatClient.h"

namespace bfs = boost::filesystem;

using namespace boost::interprocess;

Viewer::Viewer(const std::string& frame_source_name,
        std::string& save_path,
        const std::string& file_name) :

  name("viewer[" + frame_source_name + "]")
, frame_source(frame_source_name)
, min_update_period(33)
, save_path(save_path)
, file_name(file_name)
, append_date(append_date) {
    
    tick = Clock::now();
    tock = Clock::now();

    // Name *this according the the source name and the client number
    // to keep it unique
    name = name + std::to_string(frame_source.get_number_of_clients());
    cv::namedWindow(name, cv::WINDOW_NORMAL);

    // Snapshot file saving
    // First check that the save_path is valid
    bfs::path path(save_path.c_str());
    if (!bfs::exists(path) || !bfs::is_directory(path)) {
        std::cout << "Warning: requested snapshot save path, " + save_path + ", "
                << "does not exist, or is not a valid directory.\n"
                << "Using the current directory instead.\n";

        save_path = ".";
    }

    // Snapshot encoding
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
}

bool Viewer::showImage() {

    return showImage(name);
}

bool Viewer::showImage(const std::string title) {

    // If we are able to aquire the current frame, 
    // show it.
    if (frame_source.getSharedMat(current_frame)) {

        tick = Clock::now();

        milliseconds duration = std::chrono::duration_cast<milliseconds>(tick - tock);
        if (duration > min_update_period) {

            try {

                char command;

                cv::imshow(title, current_frame);
                tock = Clock::now();

                command = cv::waitKey(1);

                if (command == 's') {
                    cv::imwrite(makeFileName(), current_frame, compression_params);
                }

            } catch (cv::Exception& ex) {
                std::cerr << ex.what() << "\n";
            }
        }
    }
    
    // If server state is END, return true
    return (frame_source.getSourceRunState() == oat::ServerRunState::END);
    
}

std::string Viewer::makeFileName() {

    // Create file name
    std::time_t raw_time;
    struct tm * time_info;
    char buffer[100];
    std::time(&raw_time);
    time_info = std::localtime(&raw_time);
    std::strftime(buffer, 80, "%F-%H-%M-%S", time_info);
    std::string date_now = std::string(buffer);

    // Generate file name for this video
    if (!file_name.empty())
        frame_fid = save_path + "/" + date_now + "_" + file_name + "_" + frame_source.get_name();
    else
        frame_fid = save_path + "/" + date_now + "_" + frame_source.get_name();

    frame_fid = frame_fid + ".png";

    // Check for existence
    int i = 0;
    std::string file = frame_fid;

    while (bfs::exists(file.c_str())) {

        ++i;
        bfs::path path(frame_fid.c_str());
        bfs::path root_path = path.root_path();
        bfs::path stem = path.stem();
        bfs::path extension = path.extension();

        std::string append = "_" + std::to_string(i);
        stem += append.c_str();

        // Recreate file name
        file = std::string(root_path.generic_string()) +
                std::string(stem.generic_string()) +
                std::string(extension.generic_string());

    }

    return file;
}