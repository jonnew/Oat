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

namespace bip = boost::interprocess;

/**
 * All datatypes that will be passed through shared memory
 * must implement this ABC.
 */
class SharedObject {

    // The shared memory souce hass access to private members, but not
    // the shared memory sink
    friend SMSource;

public :

    SharedObject() { };

    SharedObject(size_t bytes, bip::managed_shared_memory::handle_t data) :
      bytes_(bytes)
    , data_(data)
    {
        // Nothing
    }

    virtual ~SharedObject() = 0;

    // This data handle is a way to pass a pointer to a mat data structure
    // through shared memory. This is important, because clients can construct
    // const cv::Mat's by getting a pointer from this handle. This means that
    // both client and server, existing in _different processes_, will use the
    // same data block for cv::Mat's data field, which is very efficient. Client
    // side constness of cv::Mat ensures copy on write behavior to prevent data
    // corruption.
    size_t bytes() const { return bytes_; }
    bip::managed_shared_memory::handle_t data() const { return data_; }

private :

    // Number of bytes to read and IP compatiable address to read them 
    cv::Size bytes_ {0};
    bip::managed_shared_memory::handle_t data_;

};

SharedObject::~SharedObject() { };

#endif	/* SHAREDOBJECT_H */
