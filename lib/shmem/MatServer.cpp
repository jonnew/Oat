/* 
 * File:   MatServer.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on April 6, 2015, 4:25 PM
 */

#include "MatServer.h"

using namespace boost::interprocess;

MatServer::MatServer(std::string server_name) :
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

    data_size = model.total() * model.elemSize();
    try {

        // Clean up any potential leftovers
        shared_memory_object::remove(shmem_name.c_str());

        // Define shared memory
        shared_memory = managed_shared_memory(open_or_create, 
                                              shmem_name.c_str(), 
                                              data_size + sizeof(shmem::SharedMatHeader) + 1024);

        // Make the shared object
        shared_mat_header = shared_memory.find_or_construct<shmem::SharedMatHeader>(shobj_name.c_str())();

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
        createSharedMat(mat);
    }
    
    scoped_lock<interprocess_sharable_mutex> lock(shared_mat_header->mutex);
    
    memcpy(shared_mat_data_ptr, mat.data, data_size);
    
    shared_mat_header->cond_var.notify_all();
    shared_mat_header->cond_var.wait(lock);
}

void MatServer::set_name(std::string server_name) {
    
    if (!shared_mat_created) {
        name = server_name;
        shmem_name = server_name + "_sh_mem";
        shobj_name = server_name + "_sh_obj";
        
    } else {
        std::cerr << "Cannot reset MatServer name after shared memory has bee allocated." << std::endl;
    }
}