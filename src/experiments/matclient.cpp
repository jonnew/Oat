//******************************************************************************
//* File:   matclient.cpp
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
 * Demo program showing very efficient shared memory passing of cv::Mat. This
 * client side program should be executed after the server side program to
 * view image that was sent to shmem. 
 */
int main(int argc, char *argv[]) {
    
    cv::namedWindow("test");
    
    // Server side
    try {
        
         // Remove shared memory on estruction, following read
        struct shm_remove
        {
           ~shm_remove(){ bip::shared_memory_object::remove("SHMEM_sh_mem"); }
        } remover;
        
        
        // Create managed shmem segment
        bip::managed_shared_memory shared_memory =
                bip::managed_shared_memory(bip::open_only, "SHMEM_sh_mem");

        // Get previously allocated shared matrix from shmem
        SharedCVMat* shmat = shared_memory.find_or_construct<SharedCVMat>("SHMAT")();

        // Get data pointer from shared handle
        void* mat_data = shared_memory.get_address_from_handle(shmat->data());

        // Construct read-only cv::Mat that uses shared memory handle to 
        // mat data
        const cv::Mat shared_mat(shmat->size(), shmat->type(), mat_data, shmat->step());
        
        cv::imshow("test", shared_mat);
        cv::waitKey(0);

    } catch (const bip::interprocess_exception& ex) {

        std::cerr << ex.what();
        return -1;
    }
    
    return 0;
}

