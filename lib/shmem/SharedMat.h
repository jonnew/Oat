/* 
 * File:   SharedMat.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on April 3, 2015, 1:48 PM
 */

#ifndef SHAREDMAT_H
#define	SHAREDMAT_H

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition_any.hpp>
#include <opencv2/core/mat.hpp>

namespace shmem {

    typedef struct {
        cv::Size size;
        int type;
        boost::interprocess::managed_shared_memory::handle_t handle;
        boost::interprocess::interprocess_sharable_mutex mutex;
        boost::interprocess::interprocess_condition_any cond_var;
    } SharedMatHeader;

//    cv::Mat constructSharedMat(SharedMat* shared_mat, boost::interprocess::managed_shared_memory* shm) {
//
//        cv::Mat mat(
//                shared_mat->size,
//                shared_mat->type,
//                shm->get_address_from_handle(shared_mat->handle));
//
//        return mat;
//    }
}


#endif	/* SHAREDMAT_H */

