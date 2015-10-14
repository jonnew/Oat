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

namespace bip = boost::interprocess;

// Class containing handle to server process's address of matrix data
class SharedCVMat {

public :
    
    SharedCVMat() { }
    
    SharedCVMat(cv::Size size, int type, bip::managed_shared_memory::handle_t data, size_t step) :
      size_(size)
    , type_(type)
    , data_(data)
    , step_(step)
    {
        // Nothing
    }
    
    // This data handle is a way to pass a pointer to a mat data structure
    // through shared memory. This is important, because clients can construct
    // const cv::Mat's by getting a pointer from this handle. This means that
    // both client and server, existing in _different processes_, will use the 
    // same data block for cv::Mat's data field, which is very efficient. Client
    // side constness of cv::Mat ensures copy on write behavior to prevent data
    // corruption.
    cv::Size size() const { return size_; }
    int type() const { return type_; }
    size_t step() const {return step_; }
    bip::managed_shared_memory::handle_t data() const { return data_; }
    
private :
    
    // Matrix metadata and handle to data
    cv::Size size_ {0, 0};
    int type_ {0};
    size_t step_ {0};
    bip::managed_shared_memory::handle_t data_;

};

#endif	/* SHAREDCVMAT_H */

