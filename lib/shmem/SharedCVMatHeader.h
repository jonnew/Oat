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

#ifndef SHAREDMAT_H
#define	SHAREDMAT_H

#include <atomic>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <opencv2/core/mat.hpp>

namespace oat {

    class SharedCVMatHeader {
    public:

        SharedCVMatHeader();

        // IPC synchronization constructs
        boost::interprocess::interprocess_semaphore mutex;
        boost::interprocess::interprocess_semaphore write_barrier;
        boost::interprocess::interprocess_semaphore read_barrier;
        boost::interprocess::interprocess_semaphore new_data_barrier;
        size_t client_read_count; // TODO: Is this used? Should it be private?

        void buildHeader(boost::interprocess::managed_shared_memory& shared_mem, const cv::Mat& model);
        void attachMatToHeader(boost::interprocess::managed_shared_memory& shared_mem, cv::Mat& mat);
        void writeSample(const uint32_t sample, const cv::Mat& value); // Server
        size_t incrementClientCount(void);
        size_t decrementClientCount(void);
        
        // Accessors
        size_t get_number_of_clients(void) const { return number_of_clients; }
        bool is_server_attached(void) const { return server_attached; }
        void set_server_attached(bool value) { server_attached = value; }
        uint32_t get_sample_number(void) const {return sample_number; }
		
    private:
        
        // Number of clients sharing this shared memory
		std::atomic<size_t> number_of_clients;

		// true when a server is attached to the shared Mat
		std::atomic<bool> server_attached;

        // Matrix primatives
        cv::Size mat_size;
        int type;
        void* data_ptr;
        int data_size_in_bytes;
        
        // Sample number
        // Should respect buffer overruns
        uint32_t sample_number;
        
        boost::interprocess::managed_shared_memory::handle_t handle;
        
    };
}

#endif	/* SHAREDMAT_H */

