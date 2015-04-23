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

#include "MatClient.h"

#include <unistd.h>
#include <boost/thread.hpp>

using namespace boost::interprocess;

MatClient::MatClient(const std::string source_name) :
name(source_name)
, shmem_name(source_name + "_sh_mem")
, shobj_name(source_name + "_sh_obj")
, shared_object_found(false)
, mat_attached_to_header(false) {
}

MatClient::MatClient(const MatClient& orig) {
}

MatClient::~MatClient() {

    detachFromShmem();
}

int MatClient::findSharedMat() {


    int client_num;
    // TODO: Wrap in a named guard of some sort.

    // Remove_shared_memory on object destruction
    //shared_memory_object::remove(shmem_name.c_str());

    try {

        // If the client creates the shared memory, it does not allocate room for the cv::Mat data
        // The server will need to resize the shared memory to make room.
        //size_t total_bytes = sizeof (shmem::SharedCVMatHeader) + 1024; 

        // TODO: This is a complete HACK until I can figure out how to resize 
        // the managed shared memory segment on the server side without 
        // causing seg faults due to bad pointers on the client side.
        size_t total_bytes = 1024e4;

        shared_memory = managed_shared_memory(open_or_create, shmem_name.c_str(), total_bytes);
        shared_mat_header = shared_memory.find_or_construct<shmem::SharedCVMatHeader>(shobj_name.c_str())();
        shared_object_found = true;

    } catch (interprocess_exception& ex) {
        std::cerr << ex.what() << '\n';
        exit(EXIT_FAILURE); // TODO: exit does not unwind the stack to take care of destructing shared memory objects
    }

    // Make sure everyone using this shared memory knows that another client
    // has joined
    shared_mat_header->mutex.wait();
    shared_mat_header->number_of_clients++;
    client_num = shared_mat_header->number_of_clients;
    shared_mat_header->mutex.post();

    return client_num;

}

void MatClient::getSharedMat(cv::Mat& value) {

    shared_mat_header->read_barrier.wait();

    /* START CRITICAL SECTION */
    shared_mat_header->mutex.wait();

    if (!mat_attached_to_header) {
        // Cannot do this until the server has called build header. 
        shared_mat_header->attachMatToHeader(shared_memory, mat);
        mat_attached_to_header = true;
    }

    value = mat.clone();

    // Now that this client has finished its read, update the count
    shared_mat_header->client_read_count++;

    // If all clients have read, signal the barrier
    if (shared_mat_header->client_read_count == shared_mat_header->number_of_clients) {
        shared_mat_header->write_barrier.post();
        shared_mat_header->client_read_count = 0;
    }

    shared_mat_header->mutex.post();
    /* END CRITICAL SECTION */

    shared_mat_header->new_data_barrier.wait();

}

void MatClient::detachFromShmem() {
    if (shared_object_found) {

        // Make sure nobody is going to wait on a disposed object
        shared_mat_header->mutex.wait();
        shared_mat_header->number_of_clients--;
        shared_mat_header->mutex.post();

#ifndef NDEBUG
        std::cout << "Number of clients in \'" + shmem_name + "\' was decremented.\n";
#endif

    }
}

void MatClient::notifySelf() {

    if (shared_object_found) {
        shared_mat_header->read_barrier.post();
        shared_mat_header->new_data_barrier.post();
    }
}

