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

#include <boost/interprocess/managed_shared_memory.hpp>
#include <opencv2/core/mat.hpp>

#include "SyncSharedMemoryObject.h"

namespace shmem {

    class SharedCVMatHeader : public SyncSharedMemoryObject {
        
    public:
        void buildHeader(boost::interprocess::managed_shared_memory& shared_mem, cv::Mat model);
        void set_value(cv::Mat mat);
        void attachMatToHeader(boost::interprocess::managed_shared_memory& shared_mem, cv::Mat& mat);

    private:
        cv::Size mat_size;
        int type;
        void* data_ptr;
        int data_size_in_bytes;
        boost::interprocess::managed_shared_memory::handle_t handle;
    };
}

#endif	/* SHAREDMAT_H */

