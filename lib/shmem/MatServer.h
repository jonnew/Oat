/* 
 * File:   MatServer.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on April 6, 2015, 4:25 PM
 */

#ifndef MATSERVER_H
#define	MATSERVER_H

#include <string>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <opencv2/core/mat.hpp>

#include "SharedMat.h"

class MatServer {
    
public:
    MatServer(std::string server_name);
    MatServer(const MatServer& orig);
    virtual ~MatServer();
    
protected:
    void createSharedMat(cv::Mat model);
    void set_shared_mat(cv::Mat mat);
    std::string name;
    bool shared_mat_created = false;

private:
    shmem::SharedMatHeader* shared_mat_header;
    void* shared_mat_data_ptr;
    int data_size; // Size of raw mat data in bytes
    
    std::string shmem_name, shobj_name, mtx_name, cond_name;
    boost::interprocess::managed_shared_memory shared_memory;

};

#endif	/* MATSERVER_H */

