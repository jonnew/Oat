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

using namespace boost::interprocess;

MatServer::MatServer(const std::string server_name) :
  name(server_name)
, shmem_name(name + "_sh_mem")
, shobj_name(name + "_sh_obj")
{ }

MatServer::MatServer(const MatServer& orig) { }

MatServer::~MatServer() {

    // Remove_shared_memory on object destruction
    shared_mat_header->cond_var.notify_all();
    shared_memory_object::remove(shmem_name.c_str());
}

void MatServer::createSharedMat(cv::Mat model) {
    
    shared_memory_object::remove(shmem_name.c_str());

    data_size = model.total() * model.elemSize();
    try {

        // Clean up any potential leftovers
        shared_memory_object::remove(shmem_name.c_str());

        // Define shared memory
        shared_memory = managed_shared_memory(open_or_create,
                shmem_name.c_str(),
                data_size + sizeof (shmem::SharedMatHeader) + 1024);

        // Make the shared object
        shared_mat_header = shared_memory.find_or_construct<shmem::SharedMatHeader>(shobj_name.c_str())();

        // Pass mutex to the scoped_lock. This will lock the shared_mat_header->mutex
        // until wait(lock) is called. This is fine because there is nothing for clients
        // to read in the shared memory until a call to set_shared_mat, which finishes
        // with wait(lock).
        //lock = makeLock();

    } catch (bad_alloc &ex) {
        std::cerr << ex.what() << '\n';
    }

    // Write the handler to the unnamed shared region holding the data
    shared_mat_data_ptr = shared_memory.allocate(data_size);

    // write the size, type and image version to the header
    shared_mat_header->size = model.size();
    shared_mat_header->type = model.type();
    shared_mat_header->handle = shared_memory.get_handle_from_address(shared_mat_data_ptr);

    shared_mat_created = true;
}

void MatServer::set_shared_mat(cv::Mat mat) {

    
    if (!shared_mat_created) {
        createSharedMat(mat); // Acquires exclusive lock on shared_mat_header->mutex
        shared_mat_header->ready = true;
    }
    
    // Exclusive scoped_lock on the shared_mat_header->mutex
    scoped_lock<interprocess_sharable_mutex> lock(shared_mat_header->mutex);
    
    // Perform write in shared memory
    memcpy(shared_mat_data_ptr, mat.data, data_size);
    
    // Notify all client processes they can now access the data
    shared_mat_header->cond_var.notify_all();
    
} // Lock is released on scope exit

//scoped_lock<interprocess_sharable_mutex> MatServer::makeLock(void) {
//    scoped_lock<interprocess_sharable_mutex> sl(shared_mat_header->mutex);
//    return sl;
//}

//void MatServer::notifyAll() {
//    
//    shared_mat_header->cond_var.notify_all();
//}
//
//void MatServer::wait() {
//    
//    shared_mat_header->cond_var.wait(lock);
//}
//
//void MatServer::notifyAllAndWait() {
//    
//    notifyAll();
//    wait();
//}