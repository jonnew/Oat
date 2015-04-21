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
//#include <boost/interprocess/sync/named_condition_any.hpp>
//#include <boost/interprocess/sync/named_sharable_mutex.hpp>
//#include <boost/interprocess/sync/named_condition.hpp>
//#include <boost/interprocess/sync/named_mutex.hpp>

using namespace boost::interprocess;

MatClient::MatClient(const std::string source_name) :
name(source_name)
, shmem_name(source_name + "_sh_mem")
, shobj_name(source_name + "_sh_obj")
, start_mutex_name(source_name + "_start_mtx")
, start_condition_name(source_name + "_start_cv")
, shared_object_found(false)
, mat_attached_to_header(false) {
}

MatClient::MatClient(const MatClient& orig) {
}

MatClient::~MatClient() {

    shared_mat_header->new_data_condition.notify_all();
}

void MatClient::findSharedMat() {

    // TODO: Wrap in a named guard of some sort.

    // Remove_shared_memory on object destruction
    shared_memory_object::remove(shmem_name.c_str());

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

    // Pass mutex to the scoped sharable_lock. 
    lock = makeLock(); // This will block until the lock has sharable access to the mutex

}

/**
 * Engages shared lock on the unnamed shared mutex residing in the shared_mat_header
 * and returns the shared cv::Mat object. The user is responsible for calling
 * notifyAndWait() following this function call, and after they are finished using the 
 * cv::Mat object to (1) release the shared lock on the mutex and (2) to puase this
 * thread until the shared cv::Mat object is updated by the server.
 * 
 * @return shared cv::Mat. Do not write on this object.
 */
cv::Mat MatClient::get_value() {

    shared_mat_header->new_data_condition.notify_all();
    shared_mat_header->new_data_condition.wait(lock);

    //    if (shared_mat_header->remap_required) {
    //        shared_memory = managed_shared_memory(open_only, shmem_name.c_str());
    //        //shared_mat_header = shared_memory.find<shmem::SharedCVMatHeader>(shobj_name.c_str())();
    //        //shared_mat_header->remap_required = false;
    //    }

    if (!mat_attached_to_header) {
        // Cannot do this until the server has called build header. 
        shared_mat_header->attachMatToHeader(shared_memory, mat);
        mat_attached_to_header = true;
    }

    return mat; // User responsible for calling notifyAndWait once they are finished processing.
}

sharable_lock<interprocess_sharable_mutex> MatClient::makeLock(void) {
    sharable_lock<interprocess_sharable_mutex> sl(shared_mat_header->mutex); // defer_lock
    return sl;
}

//void MatClient::set_source(const std::string source_name) {
//
//    // Make sure we are not already attached to some source
//    if (!shared_object_found) {
//        name = source_name;
//        shmem_name = source_name + "_sh_mem";
//        shobj_name = source_name + "_sh_obj";
//    } else {
//        std::cerr << "Cannot edit the source name because we are already reading from \"" + name + "\".";
//    }
//}

//void MatClient::notifyAndWait() {
//
//    // Wait for notification from SOURCE to grant access to shared memory
//    shared_mat_header->new_data_condition.notify_all();
//    shared_mat_header->new_data_condition.wait(lock);
//}

void MatClient::notifySelf() {

    if (shared_object_found) {
        shared_mat_header->new_data_condition.notify_one();
    }
}

