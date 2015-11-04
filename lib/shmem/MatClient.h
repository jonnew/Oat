//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
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

#ifndef MATCLIENT_H
#define	MATCLIENT_H

#include <string>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <opencv2/core/mat.hpp>

#include "SharedMemoryManager.h"
#include "SharedCVMatHeader.h"

namespace oat {

    class MatClient {
    public:
        //MatClient(void);
        MatClient(const std::string server_name);
        virtual ~MatClient();

        // get cv::Mat out of shared memory
        bool getSharedMat(cv::Mat& value);
        oat::SinkState getSourceRunState(void);

        // Accessors
        std::string get_name(void) const { return name; }
        size_t get_number_of_clients(void) const { return number_of_clients; }

        // TODO: bool is_shared_object_found(void) const { return shared_object_found; }
        uint32_t get_current_sample_number(void) const { return current_sample_number; }

    private:

        std::string name;
        oat::SharedCVMatHeader* shared_mat_header;
        oat::SharedMemoryManager* shared_mem_manager;
        bool shared_object_found, mat_attached_to_header;
        bool read_barrier_passed;
        int data_size; // Size of raw mat data in bytes

        // Shared mat object, constructed from the shared_mat_header
        cv::Mat shared_cvmat;
        const std::string shmem_name, shobj_name, shsig_name;
        boost::interprocess::managed_shared_memory shared_memory;

        // Number of clients, including *this, attached to the shared memory indicated
        // by shmem_name
        size_t number_of_clients;

        // Time keeping
        uint32_t current_sample_number;

        // Find cv::Mat object in shared memory
        int findSharedMat(void);

        // Decrement the number of clients in shared memory
        void detachFromShmem(void);
    };
} // namespace shmem

#endif	/* MATCLIENT_H */

