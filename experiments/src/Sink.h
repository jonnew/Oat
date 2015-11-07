//******************************************************************************
//* File:   Sink.h
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

#ifndef OAT_SINK_H
#define	OAT_SINK_H

#include <cassert>
#include <string>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/thread/thread_time.hpp>

#include "ForwardsDecl.h"
#include "Node.h"
#include "SharedCVMat.h"

namespace oat {

template<typename T>
class SinkBase {
public:
    SinkBase();
    virtual ~SinkBase();

    // Sinks cannot be copied
    SinkBase &operator=(const SinkBase &) = delete;
    SinkBase(const SinkBase& orig) = delete;

    void wait();
    void post();

protected:

    Node * node_ {nullptr};
    T * sh_object_ {nullptr};
    std::string address_;
    bool bound_ {false};
};

template<typename T>
SinkBase<T>::SinkBase()
{
    // Nothing
}

template<typename T>
SinkBase<T>::~SinkBase() {

    // Detach this server from shared mat header
    if (bound_) {
        node_->set_sink_state(oat::SinkState::END);

        // If the client ref count is 0, memory can be deallocated
        if (node_->source_ref_count() == 0 &&
                bip::shared_memory_object::remove(address_.c_str())) {

            // Report that shared_memory was removed on object destruction
            std::cout << "Shared memory \'" + address_ + "\' was deallocated.\n";
        }
    }
}

template<typename T>
void SinkBase<T>::wait() {
   
    // TODO: Inefficient?
    if (!bound_)
        throw("Sink must be bound before call wait() is called.");

    boost::system_time timeout = boost::get_system_time() + msec_t(10);

    // Only wait if there is a SOURCE attached to the node
    // Wait with timed wait with period check to prevent deadlocks
    while (node_->source_ref_count() > 0 &&
          !node_->write_barrier.timed_wait(timeout)) {
        // Loops checking if wait has been released
        timeout = boost::get_system_time() + msec_t(10);
    }
}

template<typename T>
void SinkBase<T>::post() {
    
    // TODO: Inefficient?
    if (!bound_)
        throw("Sink must be bound before call post() is called.");

    // Increment the number times this node has facilitated a shmem write
    node_->incrementWriteNumber();

    // Reset the source read count
    node_->resetSourceReadCount();

    // Tell each source they can read
    for (size_t i = 0; i < node_->source_ref_count(); i++)
        node_->readBarrier(i).post();
}

// Specializations...

template<typename T>
class Sink : public SinkBase<T> { 

public:
    
    void bind(const std::string &address);
    inline T& retrieve() { return *SinkBase<T>::sh_object_; };
    
private:
    shmem_t shmem_;
};

template<typename T>
void Sink<T>::bind(const std::string &address) {

    // Addresses for this block of shared memory
    SinkBase<T>::address_ = address;
    std::string node_address = SinkBase<T>::address_ + "/shmgr";
    std::string obj_address = SinkBase<T>::address_ + "/shobj";

    // Define shared memory
    // TODO: find_or_construct() will segfault if I don't provide a bit of
    //       extra space here (1024) and I don't know why
    shmem_ = bip::managed_shared_memory(
            bip::open_or_create,
            address.c_str(),
            1024 + sizeof (oat::Node) + sizeof (T));

    // Facilitates synchronized access to shmem
    SinkBase<T>::node_ = 
            shmem_.find_or_construct<oat::Node>(node_address.c_str())();

    // Make sure there is not another SINK using this shmem
    if (SinkBase<T>::node_->sink_state() != oat::SinkState::UNDEFINED) {

        // There is already a SINK using this shmem
        throw (std::runtime_error(
                "Requested SINK address, '" + 
                SinkBase<T>::address_ + "', is not available."));
    } else {

        // Find an existing shared object or construct one with default parameters
        SinkBase<T>::sh_object_ = 
                shmem_.find_or_construct<T>(obj_address.c_str())();

        SinkBase<T>::node_->set_sink_state(oat::SinkState::BOUND);
        SinkBase<T>::bound_ = true; 
    }
}

// 1. SharedCVMat

template<>
class Sink<SharedCVMat> : public SinkBase<SharedCVMat> {

public:
    void bind(const std::string &address, const size_t bytes);
    cv::Mat retrieve(const cv::Size dims, const int type);
private:
    shmem_t shmem_;

};

void Sink<SharedCVMat>::bind(const std::string &address, const size_t bytes) {
    
    // Addresses for this block of shared memory
    address_ = address;
    std::string node_address = address_ + "/shmgr";
    std::string obj_address = address_ + "/shobj";

    // Define shared memory
    shmem_ = bip::managed_shared_memory(
            bip::open_or_create,
            address.c_str(),
            1024 + bytes + sizeof(oat::Node) + sizeof(SharedCVMat));

    // Facilitates synchronized access to shmem
    node_ = shmem_.find_or_construct<oat::Node>(node_address.c_str())();

    // Make sure there is not another SINK using this shmem
    if (node_->sink_state() != oat::SinkState::UNDEFINED) {

        // There is already a SINK using this shmem
        throw (std::runtime_error(
                "Requested SINK address, '" + address_ + "', is not available."));
    } else {

        // Find an existing shared object or construct one with default parameters
        sh_object_ = shmem_.find_or_construct<SharedCVMat>(obj_address.c_str())();

        node_->set_sink_state(oat::SinkState::BOUND);
    }
}

cv::Mat Sink<SharedCVMat>::retrieve(const cv::Size dims, const int type) {

    // TODO: Would be best to grow the segment to the right size here, but in
    // order to do this, no process can be mapped to the segment. This
    // will be very hard to achieve since member objects (e.g. manager_) are
    // mapped and hang around from the lifetime of *this
    //bip::managed_shared_memory::grow(shmem_address_.c_str(), data_size);

    // Make sure that the SINK is bound to a shared memory segment
    if (!bound_)
        throw (std::runtime_error("SINK must be bound before shared cvMat is retrieved."));
    
    // Allocate memory for the shared object's data
    cv::Mat temp(dims, type);
    void *data = shmem_.allocate(temp.total() * temp.elemSize());
    handle_t handle = shmem_.get_handle_from_address(data);

    // Reset the SharedCVMat's parameters now that we know what they should be
    sh_object_->setParameters(handle, dims, type);

    // Return pointer to memory allocated for shared object
    return cv::Mat(dims, type, data);
}

} // namespace oat

#endif	/* OAT_SINK_H */
