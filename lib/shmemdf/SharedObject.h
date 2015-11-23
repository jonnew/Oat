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

#ifndef OAT_SHAREDOBJECT_H
#define	OAT_SHAREDOBJECT_H

#include <atomic>
#include <boost/interprocess/managed_shared_memory.hpp>

#include "ForwardsDecl.h"

namespace oat {

class SharedObject {

public :

    SharedObject()
    {
        // Nothing
    };

    virtual ~SharedObject()
    {
        // Nothing
    };

    // This data handle is a way to pass a pointer to a data structure
    // through shared memory. Clients can construct cotainers that use data
    // pointed by a pointer from this handle. This means that
    // both client and server, existing in _different processes_, will have direct
    // access to this data structure
    handle_t sample() const { return sample_; }
    handle_t data() const { return data_; }

protected :

    std::atomic<handle_t> data_;
    std::atomic<handle_t> sample_;
};

} // namepace oat

#endif	/* OAT_SHAREDOBJECT_H */
