/* 
 * File:   MatClient.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on April 6, 2015, 8:47 PM
 */

#include "MatClient.h"

using namespace boost::interprocess;

MatClient::MatClient(std::string server_name) :
name(server_name)
, shmem_name(server_name.append("_sh_mem"))
, shobj_name(server_name.append("_sh_obj")) {
}

MatClient::MatClient(const MatClient& orig) {
}

MatClient::~MatClient() {

    // Clean up sync objects
    shared_mat_header->cond_var.notify_all();
}

void MatClient::findSharedMat() {

    shared_memory = managed_shared_memory(open_only, shmem_name.c_str());
    shared_mat_header = shared_memory.find<shmem::SharedMatHeader>(shobj_name.c_str()).first;

    shared_mat_created = true;
    
    mat.create(shared_mat_header->size,
               shared_mat_header->type);
    
    mat.data = (uchar*)shared_memory.get_address_from_handle(shared_mat_header->handle);

}

cv::Mat MatClient::get_shared_mat() {

    return mat;
}
