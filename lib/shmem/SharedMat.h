/* 
 * File:   SharedMat.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on April 3, 2015, 1:48 PM
 */

#ifndef SHAREDMAT_H
#define	SHAREDMAT_H

#include <boost/interprocess/managed_shared_memory.hpp>
#include <opencv2/core/mat.hpp>

namespace sharedmat {

    typedef struct {
        cv::Size size;
        int type;
        boost::interprocess::managed_shared_memory::handle_t handle;
    } SharedMat;

    cv::Mat constructMat(SharedMat shared_mat) {

        cv::Mat mat(
            shared_mat->size,
            shared_mat->type,
            shared_mat.get_address_from_handle(shared_mat->handle));

        return mat;
    }
}


#endif	/* SHAREDMAT_H */

