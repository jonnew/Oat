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

#ifndef SHAREDMAT_H
#define	SHAREDMAT_H

#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <opencv2/core/mat.hpp>

namespace shmem {

    class SharedCVMatHeader {
        
    public:
        
        SharedCVMatHeader();
        
        boost::interprocess::interprocess_semaphore mutex;
        boost::interprocess::interprocess_semaphore write_barrier;
        boost::interprocess::interprocess_semaphore read_barrier;
        boost::interprocess::interprocess_semaphore new_data_barrier;
        
        size_t number_of_clients;
        size_t client_read_count;
        
        // Used to get world coordinates from image
        // TODO: Replace with homography transformation matrix
        bool world_coords_valid;
        cv::Point2f xy_origin_in_px;
        float worldunits_per_px_x;
        float worldunits_per_px_y;

        void buildHeader(boost::interprocess::managed_shared_memory& shared_mem, const cv::Mat& model);
        void attachMatToHeader(boost::interprocess::managed_shared_memory& shared_mem, cv::Mat& mat);
        
        // Accessors
        void set_value(const cv::Mat& mat);    // Server
        
    private:
        
        cv::Size mat_size;
        int type;
        void* data_ptr;
        int data_size_in_bytes;
        
        boost::interprocess::managed_shared_memory::handle_t handle;
    };
}

#endif	/* SHAREDMAT_H */

