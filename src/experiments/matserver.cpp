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
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/program_options.hpp>

#include "SharedCVMat.h"

namespace bip = boost::interprocess;

/*
 * 
 */
int main(int argc, char *argv[]) {

    // image file to use to populate cv_mat
    std::string file_name = "/home/jon/Desktop/test.png";
    
    // Read file to get sample image (simulates a camera grab)
    cv::Mat ext_mat = cv::imread(file_name);

    // Server side
    try {
        
        // Remove shared memory on construction
        // Client will remove on destruction
        struct shm_remove
        {
           shm_remove() { bip::shared_memory_object::remove("SHMEM_sh_mem"); }
        } remover;
        
        
        // Create managed shmem segment
        bip::managed_shared_memory shared_memory =
                bip::managed_shared_memory(bip::create_only, "SHMEM_sh_mem", 10e6);

        // How many bytes per matrix?
        cv::Size mat_dims(ext_mat.cols, ext_mat.rows);
        size_t step = ext_mat.step[0];
        size_t mat_size = step * ext_mat.rows;
        
        // Allocate mat_size bytes in shmem segment to hold mat data
        void* mat_data = shared_memory.allocate(mat_size);

        // Create the shared cv::Mat that uses shem to store data 
        cv::Mat shared_mat(mat_dims, ext_mat.type(), mat_data, step);
        
        // This should not be needed normally -- images from a camera or video file
        // should just be put directly into the space pointed to by mat_data.
        memcpy(mat_data, ext_mat.data, mat_size);

        // Create a cross-process compatible handle to the shmem address
        // holding mat data
        bip::managed_shared_memory::handle_t handle =
                shared_memory.get_handle_from_address(mat_data);

        // Created shared memory-based cv::Mat
        SharedCVMat* shmat = shared_memory.construct<SharedCVMat>
                ("SHMAT")(mat_dims, shared_mat.type(), handle, step);

    } catch (const bip::interprocess_exception& ex) {

        std::cerr << ex.what();
        return -1;
    }
    
    return 0;
}

