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

#ifndef BUFFEREDMATSERVER_H
#define	BUFFEREDMATSERVER_H

#include <atomic>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <opencv2/core/mat.hpp>

#include "SharedCVMatHeader.h"
#include "SharedMemoryManager.h"

namespace oat {

    // TODO: Find a why to integrate this with the must much general purpose SMServer
    class BufferedMatServer {
    public:
        BufferedMatServer(const std::string& sink_name);
        BufferedMatServer(const BufferedMatServer& orig);
        virtual ~BufferedMatServer();

        void pushMat(const cv::Mat& mat, const uint32_t& sample_number);
        void setSharedServerState(oat::ServerRunState state);
        
        // Accessors 
        std::string get_name(void) const { return name; }
        void set_running(bool value) {serve_thread_running = value; }

    private:

        // Name of this server
        std::string name;

        // Buffer
        static const int MATSERVER_BUFFER_SIZE = 1024;
        boost::lockfree::spsc_queue
        <std::pair<unsigned int, cv::Mat>, boost::lockfree::capacity<MATSERVER_BUFFER_SIZE> > mat_buffer;

        // Server threading
        std::thread server_thread;
        std::mutex server_mutex;
        std::atomic<bool> serve_thread_running;
        std::condition_variable serve_condition;
        oat::SharedCVMatHeader* shared_mat_header;
        oat::SharedMemoryManager* shared_mem_manager;
        bool shared_object_created;
        bool mat_header_constructed;

        int data_size; // Size of raw mat data in bytes

        const std::string shmem_name, shobj_name, shmgr_name;
        boost::interprocess::managed_shared_memory shared_memory;
        
        void createSharedMat(void);

        /**
         * Synchronized shared memory publication.
         * @param mat
         */
        void serveMatFromBuffer(void);

        /**
         * Auto post to this servers own semaphore.wait() to allow threads to unblock
         * in order to ensure proper object destruction.
         * @param mat
         */
        void notifySelf(void);

#ifndef NDEBUG
        const int BAR_WIDTH = 50;
#endif

    };

} 

#endif	/* BUFFEREDMATSERVER_H */

