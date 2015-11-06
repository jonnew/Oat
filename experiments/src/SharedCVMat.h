//******************************************************************************
//* File:   SharedCVMat.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
//* All right reserved.
//* This file is part of the Oat project.
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

#ifndef SHAREDCVMAT_H
#define	SHAREDCVMAT_H

#include <atomic>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <opencv2/core.hpp>

#include "SharedObject.h"

namespace oat {
namespace bip = boost::interprocess;

// Class containing handle to server process's address of matrix data
class SharedCVMat : public SharedObject{
    
    using handle_t = bip::managed_shared_memory::handle_t;
    
public :
    
    SharedCVMat() : SharedObject() { }

    cv::Size size() const { return size_; }
    int type() const { return type_; }
    size_t step() const {return step_; }
    
    void setParameters(const handle_t data, const cv::Size size, const int type) { 
        data_ = data;
        size_ = size;
        type_ = type;
    }
    
private :
    
    // Matrix metadata and handle to data
    cv::Size size_ {0, 0}; // Should be atomic...
    std::atomic<int> type_ {0};
    std::atomic<size_t> step_ {0};

};

}

#endif	/* SHAREDCVMAT_H */

