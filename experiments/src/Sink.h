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

#include <string>
#include <memory>
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

    shmem_t node_shmem_, obj_shmem_;
    Node * node_ {nullptr};
    T * sh_object_ {nullptr};
    std::string node_address_, obj_address_;
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
        node_->set_sink_state(SinkState::END);

        // If the client ref count is 0, memory can be deallocated
        if (node_->source_ref_count() == 0 &&
            bip::shared_memory_object::remove(node_address_.c_str()) &&
            bip::shared_memory_object::remove(obj_address_.c_str())) {

#ifndef NDEBUG
        std::cout << "Shared memory at \'" + node_address_ +
                "\' and \'" + obj_address_ + "\' was deallocated.\n";
#endif
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
    // TODO: This is wrong. What if source 0 connects, then source 1, then source 0
    // disconnects. Then we are posting to the wrong source...
    for (size_t i = 0; i < node_->source_ref_count(); i++)
        node_->readBarrier(i).post();
}

// Specializations...

template<typename T>
class Sink : public SinkBase<T> {

    using SinkBase<T>::node_address_;
    using SinkBase<T>::obj_address_;
    using SinkBase<T>::node_shmem_;
    using SinkBase<T>::obj_shmem_;
    using SinkBase<T>::node_;
    using SinkBase<T>::sh_object_;
    using SinkBase<T>::bound_;

public:

    void bind(const std::string &address);
    T * retrieve();
};

template<typename T>
void Sink<T>::bind(const std::string &address) {

    // Addresses for this block of shared memory
    node_address_ = address + "_node";
    obj_address_ = address + "_obj";

    // Define shared memory
    // TODO: find_or_construct() will segfault if I don't provide a bit of
    //       extra space here (1024) and I don't know why
    node_shmem_ = bip::managed_shared_memory(
            bip::open_or_create,
            node_address_.c_str(),
            1024 + sizeof(Node));

    // Bind to a node which facilitates synchronized access to shmem
    node_ = node_shmem_.template find_or_construct<Node>(typeid(Node).name())();

    // Make sure there is not another SINK using this shmem
    if (node_->sink_state() != SinkState::UNDEFINED) {

        // There is already a SINK using this shmem
        throw (std::runtime_error(
                "Requested SINK address, '" + address + "', is not available."));
    } else {

        obj_shmem_ = bip::managed_shared_memory(
            bip::create_only,
            obj_address_.c_str(),
            1024 + sizeof (T));

        // Find an existing shared object or construct one
        sh_object_ = obj_shmem_.template find_or_construct<T>(typeid(T).name())();
        node_->set_sink_state(SinkState::BOUND);
        bound_ = true;
    }
}

template<typename T>
T * Sink<T>::retrieve() {

    //assert(SinkBase<T>::bound_);
    if (!bound_)
        throw (std::runtime_error("SINK must be bound before shared object is retrieved."));

    return sh_object_;
}

// 1. SharedCVMat

template<>
class Sink<SharedCVMat> : public SinkBase<SharedCVMat> {

public:
    void bind(const std::string &address, const size_t bytes);
    cv::Mat retrieve(const cv::Size dims, const int type);
};

void Sink<SharedCVMat>::bind(const std::string &address, const size_t bytes) {

    // Addresses for this block of shared memory
    node_address_ = address + "_node";
    obj_address_ = address + "_obj";

    // Define shared memory
    node_shmem_ = bip::managed_shared_memory(
            bip::open_or_create,
            node_address_.c_str(),
            1024  + sizeof(Node));

    // Facilitates synchronized access to shmem
    node_ = node_shmem_.find_or_construct<Node>(typeid(Node).name())();

    // Make sure there is not another SINK using this shmem
    if (node_->sink_state() != SinkState::UNDEFINED) {

        // There is already a SINK using this shmem
        throw (std::runtime_error(
                "Requested SINK address, '" + address + "', is not available."));
    } else {

        obj_shmem_ = bip::managed_shared_memory(
            bip::create_only,
            obj_address_.c_str(),
            1024 + sizeof(SharedCVMat) + bytes);

        // Find an existing shared object or construct one
        sh_object_ = obj_shmem_.find_or_construct<SharedCVMat>(typeid(SharedCVMat).name())();

        node_->set_sink_state(SinkState::BOUND);
        bound_ = true;
    }
}

cv::Mat Sink<SharedCVMat>::retrieve(const cv::Size dims, const int type) {

    // Make sure that the SINK is bound to a shared memory segment
    //assert(bound_);
    if (!bound_)
        throw (std::runtime_error("SINK must be bound before shared cvMat is retrieved."));

    // Allocate memory for the shared object's data
    cv::Mat temp(dims, type);
    void *data = obj_shmem_.allocate(temp.total() * temp.elemSize());
    handle_t handle = obj_shmem_.get_handle_from_address(data);

    // Reset the SharedCVMat's parameters now that we know what they should be
    sh_object_->setParameters(handle, dims, type);

    // Return pointer to memory allocated for shared object
    return cv::Mat(dims, type, data);
}

} // namespace oat

#endif	/* OAT_SINK_H */
