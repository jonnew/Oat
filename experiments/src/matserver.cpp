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

#include <csignal>
#include <exception>

#include "SharedCVMat.h"
#include "NodeManager.h"
#include "Sink.h"
#include "SharedCVMat.h"


volatile sig_atomic_t quit = 0;

// Signal handler to ensure shared resources are cleaned on exit due to ctrl-c
void sigHandler(int s) {
    quit = 1;
}

/*
 * Demo program showing very efficient shared memory passing of cv::Mat. This
 * server side program should be executed first to load data into shmem.
 */
int main(int argc, char *argv[]) {
    
    std::signal(SIGINT, sigHandler);

    // Image to send through shmem
    // Change to some sample image on your filesystem.
    std::string file_name = "/home/jon/Desktop/test.png";

    try {

        // Read file to get sample image (simulates a camera grab)
        cv::Mat ext_mat = cv::imread(file_name);

        // How many bytes per matrix?
        cv::Size mat_dims(ext_mat.cols, ext_mat.rows);
        size_t step = ext_mat.step[0];
        size_t mat_size = step * ext_mat.rows;
        
        // Create sink to send matrix into
        oat::Sink<oat::SharedCVMat> sink;
        sink.bind("exp_sh_mem", 10e6);
        void* mat_data = sink.allocate(mat_size, mat_dims, ext_mat.type(), step);
        
        // Create the shared cv::Mat that uses shem to store data 
        cv::Mat shared_mat(mat_dims, ext_mat.type(), mat_data, step);
        
        // This should not be needed normally -- images from a camera or video file
        // should just be put directly into the space pointed to by mat_data.
        memcpy(mat_data, ext_mat.data, mat_size);

        uint64_t angle = 0;
        while(!quit) {
            
            cv::Point2f src_center(ext_mat.cols/2.0F, ext_mat.rows/2.0F);
            cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, ++angle, 1.0);
            
            // Start critical section
            cv::warpAffine(ext_mat, shared_mat, rot_mat, ext_mat.size());
            // End critical section
            
            std::cout << "Sample: " << angle << "\r";
            std::cout.flush();

        }

    } catch (const std::exception& ex) {

        std::cerr << ex.what();
        return -1;
    }
    
    return 0;
}

