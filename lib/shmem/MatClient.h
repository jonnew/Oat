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

#ifndef MATCLIENT_H
#define	MATCLIENT_H

#include <string>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <opencv2/core/mat.hpp>

#include "SharedCVMatHeader.h"

namespace shmem {

    class MatClient {
    public:
        //MatClient(void);
        MatClient(const std::string server_name);
        virtual ~MatClient();

        // get cv::Mat out of shared memory
        bool getSharedMat(cv::Mat& value);

        // Accessors

        std::string get_name(void) {
            return name;
        }

        int get_number_of_clients(void) {
            return number_of_clients;
        }

        bool is_homography_valid(void) {
            return shared_mat_header->homography_valid;
        }

        cv::Matx33d get_homography(void) {
            return shared_mat_header->homography;
        }

        bool is_shared_object_found(void) {
            return shared_object_found;
        }

        uint32_t get_current_time_stamp(void) {
            return current_time_stamp;
        }

    private:

        std::string name;
        shmem::SharedCVMatHeader* shared_mat_header;
        bool shared_object_found, mat_attached_to_header;
        bool read_barrier_passed;
        int data_size; // Size of raw mat data in bytes

        // Shared mat object, constructed from the shared_mat_header
        cv::Mat shared_cvmat;
        const std::string shmem_name, shobj_name;
        boost::interprocess::managed_shared_memory shared_memory;

        // Number of clients, including *this, attached to the shared memory indicated
        // by shmem_name
        size_t number_of_clients;

        // Time keeping
        uint32_t current_time_stamp;

        // Find cv::Mat object in shared memory
        void findSharedMat(void);

        // Decrement the number of clients in shared memory
        void detachFromShmem(void);
    };
} // namespace shmem

#endif	/* MATCLIENT_H */

