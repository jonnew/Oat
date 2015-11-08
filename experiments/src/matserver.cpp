//******************************************************************************
//* File:   main.cpp
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
//******************************************************************************

#include <opencv2/opencv.hpp>

#include <chrono>
#include <csignal>
#include <exception>

#include "SharedCVMat.h"
#include "Node.h"
#include "Sink.h"
#include "SharedCVMat.h"

using Clock = std::chrono::high_resolution_clock;
using Milliseconds = std::chrono::milliseconds;

Clock::time_point tick = Clock::now();
Clock::time_point tock = Clock::now();

volatile sig_atomic_t quit = 0;

// Signal handler to ensure shared resources are cleaned on exit due to ctrl-c
void sigHandler(int s) {
    quit = 1;
}

/*
 * Demo program showing efficient shared memory passing of cv::Mat. This
 * server side program should be executed first to load data into shmem.
 */
int main(int argc, char *argv[]) {

    std::signal(SIGINT, sigHandler);

    char const * file;
    if (argc == 1)
        file = "/home/jon/Desktop/test.png";
    else if (argc == 2)
        file = argv[1];
    else {
        std::cerr << "Usage: oat-exp-server <path to image>\n";
        return -1;
    }

    // Image to send through shmem
    std::string file_name {file};

    try {

        // Read file to get sample image (simulates a camera grab)
        cv::Mat ext_mat = cv::imread(file_name);
        if (ext_mat.empty()) {
            std::cerr << "Image path invalid.\n";
            return -1;
        }

        // How many bytes per matrix?
        cv::Size mat_dims(ext_mat.cols, ext_mat.rows);

        // Create sink to send matrix into
        oat::Sink<oat::SharedCVMat> sink;
        sink.bind("exp", 10e6);

        cv::Mat shared_mat = sink.retrieve(mat_dims, ext_mat.type());

        uint64_t angle = 0;
        cv::Point2f src_center(ext_mat.cols/2.0F, ext_mat.rows/2.0F);
        cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, ++angle, 1.0);
        while(!quit) {

            tick = Clock::now();

            Milliseconds duration =
                std::chrono::duration_cast<Milliseconds>(tick - tock);

            // Do transform
            cv::warpAffine(ext_mat, shared_mat, rot_mat, ext_mat.size());
            tock = Clock::now();

            // Tell sources there is new data
            sink.post();

            // Get new rotation matrix
            rot_mat = cv::getRotationMatrix2D(src_center, ++angle, 1.0);

            std::cout << "Loop duration (ms): " << duration.count() << "\r";
            std::cout.flush();

            // Wait for sources to finish reading
            sink.wait();
        }

    } catch (const std::exception& ex) {

        std::cerr << ex.what();
        return -1;
    }

    return 0;
}

