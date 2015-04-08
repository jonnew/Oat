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
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition_any.hpp>
#include <opencv2/core/mat.hpp>

namespace shmem {

    typedef struct {
        bool ready = false;
        cv::Size size;
        int type;
        boost::interprocess::managed_shared_memory::handle_t handle;
        boost::interprocess::interprocess_sharable_mutex mutex;
        boost::interprocess::interprocess_condition_any cond_var;
    } SharedMatHeader;
}


#endif	/* SHAREDMAT_H */

