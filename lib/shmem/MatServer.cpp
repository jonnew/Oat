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

#include "MatServer.h"

#include <chrono>
#include <boost/interprocess/managed_shared_memory.hpp>

#include "SharedCVMatHeader.h"
#include "SharedCVMatHeader.cpp" // TODO: Why???

using namespace boost::interprocess;

MatServer::MatServer(const std::string& sink_name) :
name(sink_name)
, shmem_name(sink_name + "_sh_mem")
, shobj_name(sink_name + "_sh_obj")
, shared_object_created(false)
, current_sample(0)
, write_index(0)
, running(true) {

    // Start the server thread
    server_thread = std::thread(&MatServer::serveMatFromBuffer, this);
}

MatServer::MatServer(const MatServer& orig) {
}

MatServer::~MatServer() {

    running = false;

    // Make sure we unblock the server thread
    for (int i = 0; i <= MATSERVER_BUFFER_SIZE; ++i) {
        notifySelf();
    }

    // Join the server thread back with the main one
    server_thread.join();

    // Remove_shared_memory on object destruction

    shared_memory_object::remove(shmem_name.c_str());
#ifndef NDEBUG
    std::cout << "Shared memory \'" + shmem_name + "\' was deallocated.\n";
#endif

}

void MatServer::createSharedMat(const cv::Mat& model) {

    data_size = model.total() * model.elemSize();

    // TODO: Wrap in a named guard of some sort to protect the grow operation, which
    // is not thread safe
    try {

        // Total amount of shared memory to allocated
        //size_t total_bytes = data_size + sizeof (shmem::SharedCVMatHeader) + 1024;

        // TODO: This is a complete HACK until I can figure out how to resize 
        // the managed shared memory segment on the server side without 
        // causing seg faults due to bad pointers on the client side.
        size_t total_bytes = 1024e4;

        // Define shared memory
        shared_memory = managed_shared_memory(open_or_create,
                shmem_name.c_str(),
                total_bytes);

        shared_mat_header = shared_memory.find_or_construct<shmem::SharedCVMatHeader>(shobj_name.c_str())();

        // Check if client allocated, in which case we need to make room for
        // the cv::Mat data
        //        if (shared_memory.get_size() < total_bytes) {
        //            size_t extra_bytes =  total_bytes - shared_memory.get_size();
        //            managed_shared_memory::grow(shmem_name.c_str(), extra_bytes);
        //            shared_memory = managed_shared_memory(open_only, shmem_name.c_str());
        //            shared_mat_header = shared_memory.find_or_construct<shmem::SharedCVMatHeader>(shobj_name.c_str())();
        //            shared_mat_header->remap_required = true;
        //        } 
        //        else {
        //            shared_mat_header = shared_memory.find_or_construct<shmem::SharedCVMatHeader>(shobj_name.c_str())();
        //        }

    } catch (interprocess_exception &ex) {
        std::cerr << ex.what() << '\n';
        exit(EXIT_FAILURE); // TODO: exit does not unwind the stack to take care of destructing shared memory objects
    }

    shared_mat_header->buildHeader(shared_memory, model);

    // If supplied with valid homography info, then set that
    if (homography_valid) {
        shared_mat_header->homography_valid = true;
        shared_mat_header->homography = homography;
    }

    shared_object_created = true;
}

void MatServer::pushMat(const cv::Mat& mat) {

    // Push data onto ring buffer
    mat_buffer.push(mat.clone());
    tick_buffer.push(current_sample++);

#ifndef NDEBUG
    std::cout << "Buffer count: " + std::to_string(mat_buffer.read_available())
              << ". Sample no. : " + std::to_string(current_sample - 1) + "\n";
#endif

    // notify server thread that data is available
    serve_condition.notify_one();
}

void MatServer::serveMatFromBuffer() {

    while (running) {

        // Proceed only if mat_buffer has data
        std::unique_lock<std::mutex> lk(server_mutex);
        serve_condition.wait_for(lk, std::chrono::milliseconds(10));

        // Here we must attempt to clear the whole buffer before waiting again.
        cv::Mat mat;
        while (mat_buffer.pop(mat) && running) {

            // Create shared mat object if not done already
            if (!shared_object_created) {
                createSharedMat(mat);
            }

            /* START CRITICAL SECTION */
            shared_mat_header->mutex.wait();

            // Perform writes in shared memory 
            shared_mat_header->set_mat(mat);
            tick_buffer.pop(shared_mat_header->sample_number); 
            shared_mat_header->sample_index = write_index++;

            // Tell each client they can proceed
            for (int i = 0; i < shared_mat_header->number_of_clients; ++i) {
                shared_mat_header->read_barrier.post();
            }

            shared_mat_header->mutex.post();
            /* END CRITICAL SECTION */

            // Only wait if there is a client
            if (shared_mat_header->number_of_clients) {
                shared_mat_header->write_barrier.wait();
            }

            // Tell each client they can proceed now that the write_barrier
            // has been passed
            for (int i = 0; i < shared_mat_header->number_of_clients; ++i) {
                shared_mat_header->new_data_barrier.post();
            }
        }
    }
}

void MatServer::notifySelf() {

    if (shared_object_created) {
        shared_mat_header->write_barrier.post();
    }
}