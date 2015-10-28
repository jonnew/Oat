//******************************************************************************
//* File:   SharedObject.h
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

#ifndef SHAREDOBJECT_H
#define	SHAREDOBJECT_H

#include <atomic>
#include <boost/interprocess/managed_shared_memory.hpp>

namespace oat {
namespace bip = boost::interprocess;
/**
 * All datatypes that will be passed through shared memory
 * must implement this ABC.
 */
class SharedObject {

    using handle_t = bip::managed_shared_memory::handle_t;
    
public :

    SharedObject() { };

    SharedObject(handle_t data) : //size_t bytes, 
      //bytes_(bytes)
      data_(data)
    {
        // Nothing
    }

    virtual ~SharedObject() = 0;
    
    template<typename... Ts>
    //virtual void setParameters(Ts...) = 0;

    // This data handle is a way to pass a pointer to a mat data structure
    // through shared memory. This is important, because clients can construct
    // const cv::Mat's by getting a pointer from this handle. This means that
    // both client and server, existing in _different processes_, will use the
    // same data block for cv::Mat's data field, which is very efficient. Client
    // side constness of cv::Mat ensures copy on write behavior to prevent data
    // corruption.
    //size_t bytes() const { return bytes_; }
    handle_t data() const { return data_; }
    
protected :

    // Number of bytes to read and IP compatable address to read them 
    //std::atomic<size_t> bytes_ {0};
    std::atomic<handle_t> data_;

};

SharedObject::~SharedObject() { };

} // namepace oat

#endif	/* SHAREDOBJECT_H */
