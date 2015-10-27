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

#include "BufferedMatServer.h"

#include <ostream>
#include <chrono>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/thread/thread_time.hpp>

#include "../../lib/utility/IOFormat.h"
#include "SharedCVMatHeader.cpp" // TODO: Why???

namespace oat {

    namespace bip = boost::interprocess;

    BufferedMatServer::BufferedMatServer(const std::string& sink_name) :
      name(sink_name)
    , serve_thread_running(true)
    , shmem_name(sink_name + "_sh_mem")
    , shobj_name(sink_name + "_sh_obj")
    , shmgr_name(sink_name + "_sh_mgr")
    , shared_object_created(false)
    , mat_header_constructed(false) {
        
        // Create shared mat first so that server thread has something to play
        // with
        createSharedMat();

        // Start the server thread
        server_thread = std::thread(&BufferedMatServer::serveMatFromBuffer, this);

    }

    BufferedMatServer::BufferedMatServer(const BufferedMatServer& orig) {
    }

    BufferedMatServer::~BufferedMatServer() {

        serve_thread_running = false;
        
        // Make sure we unblock the server thread
        for (int i = 0; i <= MATSERVER_BUFFER_SIZE; ++i) {
            notifySelf();
        }

        // Join the server thread back with the main one
        server_thread.join();
        
        // Set stream EOF state in shmem
        shared_mem_manager->set_server_state(oat::SinkState::END);

        // TODO: If the client ref count is 0, memory can be deallocated
        if (shared_mem_manager->source_ref_count() == 0) {
            
            // Remove_shared_memory on object destruction
            bip::shared_memory_object::remove(shmem_name.c_str());
#ifndef NDEBUG
            std::cout << oat::dbgMessage("Shared memory \'" + shmem_name + "\' was deallocated.\n");
#endif
        } 
    }

    void BufferedMatServer::createSharedMat(void) {
        
        // TODO: I am currently using a static 10 MB block to store shared
        // cv::Mat headers and data. This is a bit of a hack  until 
        // I can figure out how to resize the managed shared memory segment 
        // on the server side without causing seg faults due to bad pointers on the client side.

        // Total amount of shared memory to allocated
        size_t total_bytes = 1024e4;

        // Define shared memory
        shared_memory = bip::managed_shared_memory(bip::open_or_create,
                shmem_name.c_str(),
                total_bytes);

        shared_mat_header = shared_memory.find_or_construct<oat::SharedCVMatHeader>(shobj_name.c_str())();
        shared_mem_manager = shared_memory.find_or_construct<oat::NodeManger>(shmgr_name.c_str())();

        // Make sure there is not another server using this shmem
        if (shared_mem_manager->get_server_state() != oat::SinkState::UNDEFINED) {

            // There is already a server using this shmem
            throw (std::runtime_error(
                    "Requested SINK name, '" + name + "', is not available."));

        } else {

            shared_object_created = true;
            shared_mem_manager->set_server_state(oat::SinkState::BOUND);
        }
    }

    /**
     * Push a deep copy of cv::Mat object to shared memory along with sample number.
     * @param mat cv::Mat to push to shared memory
     * @param sample_number sample number of cv::Mat
     */
    void BufferedMatServer::pushMat(const cv::Mat& mat, const uint32_t& sample_number) {

        // Push data onto ring buffer
        mat_buffer.push(std::make_pair(sample_number, mat.clone()));

        // notify server thread that data is available
        serve_condition.notify_one();
    }

    void BufferedMatServer::serveMatFromBuffer() {

        while (serve_thread_running) {

            // Proceed only if mat_buffer has data
            std::unique_lock<std::mutex> lk(server_mutex);
            serve_condition.wait_for(lk, std::chrono::milliseconds(10));

            // Here we must attempt to clear the whole buffer before waiting again.
            std::pair<uint32_t, cv::Mat> sample;

            boost::system_time timeout =
                    boost::get_system_time() + boost::posix_time::milliseconds(10);
            
            while (mat_buffer.pop(sample)) {

#ifndef NDEBUG

                std::cout << oat::dbgMessage("[");

                int progress = (BAR_WIDTH * mat_buffer.read_available()) / MATSERVER_BUFFER_SIZE;
                int remaining = BAR_WIDTH - progress;

                for (int i = 0; i < progress; ++i) {
                    std::cout << oat::dbgColor("=");
                }
                for (int i = 0; i < remaining; ++i) {
                    std::cout << " ";
                }
                
                std::cout << oat::dbgColor("] ")
                        << oat::dbgColor(std::to_string(mat_buffer.read_available()) + "/" + std::to_string(MATSERVER_BUFFER_SIZE))
                        << oat::dbgColor(", sample: " + std::to_string(sample.first))
                        << "\r";

                std::cout.flush();

#endif
                try {
                    // Create shared mat object if not done already
                    if (!mat_header_constructed) {
                        shared_mat_header->buildHeader(shared_memory, sample.second);
                        mat_header_constructed = true;
                    }

                    /* START CRITICAL SECTION */
                    shared_mat_header->mutex.wait();

                    // Perform writes in shared memory 
                    shared_mat_header->writeSample(sample.first, sample.second);

                    // Tell each client they can proceed
                    for (int i = 0; i < shared_mem_manager->source_ref_count(); ++i) {
                        shared_mat_header->read_barrier.post();
                    }

                    shared_mat_header->mutex.post();
                    /* END CRITICAL SECTION */

                    // Only wait if there is a client
                    // Timed wait with period check to prevent deadlocks
                    while (shared_mem_manager->source_ref_count() > 0 &&
                            !shared_mat_header->write_barrier.timed_wait(timeout)) {
                    }


                    // Tell each client they can proceed now that the write_barrier
                    // has been passed
                    for (int i = 0; i < shared_mem_manager->source_ref_count(); ++i) {
                        shared_mat_header->new_data_barrier.post();
                    }
                } catch (bip::interprocess_exception ex) {

                    // Something went wrong during shmem access so result is invalid
                    // Usually due to SIGINT being called during read_barrier timed
                    // wait
                    return;
                }
            }
        }
    }
 
    void BufferedMatServer::notifySelf() {

        if (shared_object_created) {
            shared_mat_header->write_barrier.post();
        }
    }
}
