/* 
 * File:   MatClient.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on April 6, 2015, 8:47 PM
 */

#ifndef MATCLIENT_H
#define	MATCLIENT_H

#include "SharedMat.h"
#include <string>

class MatClient {
public:
    MatClient(std::string server_name);
    MatClient(const MatClient& orig);
    virtual ~MatClient();
    
protected:
    void findSharedMat(void);
    cv::Mat get_shared_mat(void);
    void set_name(std::string server_name);
    std::string name;
    shmem::SharedMatHeader* shared_mat_header;
    bool shared_mat_created = false;

private:
    
    cv::Mat mat;

    void* shared_mat_data_ptr;
    int data_size; // Size of raw mat data in bytes
    
    std::string shmem_name, shobj_name, mtx_name, cond_name;
    boost::interprocess::managed_shared_memory shared_memory;

};

#endif	/* MATCLIENT_H */

