//******************************************************************************
//* File:   Source.h
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

#ifndef OAT_SOURCE_H
#define	OAT_SOURCE_H

#include <iostream>
#include <string>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/thread/thread_time.hpp>

#include "../datatypes/Frame.h"

#include "ForwardsDecl.h"
#include "Node.h"
#include "SharedCVMat.h"

namespace oat {

template<typename T>
class SourceBase  {
public:
    SourceBase();
    virtual ~SourceBase();

    virtual void connect(const std::string &address);
    NodeState wait();
    void post();

    uint64_t write_number() const {
        return (node_ == nullptr ? 0 : node_->write_number());
    }

protected:

    shmem_t node_shmem_, obj_shmem_;
    T * sh_object_ {nullptr};
    Node * node_ {nullptr};
    std::string node_address_, obj_address_;
    size_t slot_index_;
    bool bound_ {false};
    bool connected_ {false};
    bool did_wait_need_post_ {false};
};

template<typename T>
inline SourceBase<T>::SourceBase()
{
    // Nothing
}

template<typename T>
inline SourceBase<T>::~SourceBase() {

    // If the client reference count is 0 and there is no server
    // attached to the node, deallocate the shmem
    if (bound_ &&
        node_->releaseSlot(slot_index_) == 0 &&
        node_->sink_state() != NodeState::SINK_BOUND &&
        bip::shared_memory_object::remove(node_address_.c_str()) &&
        bip::shared_memory_object::remove(obj_address_.c_str())) {

#ifndef NDEBUG
        std::cout << "Shared memory at \'" + node_address_ +
                "\' and \'" + obj_address_ + "\' was deallocated.\n";
#endif
    }
}

template<typename T>
inline void SourceBase<T>::connect(const std::string &address) {

    if (connected_ || bound_)
        throw std::runtime_error("A source can only connect a "
                                 "single time to a single node.");

    // Addresses for this block of shared memory
    node_address_ = address + "_node";
    obj_address_ = address + "_obj";

    // Define shared memory
    // Extra 1024 bytes are used to hold managed shared mem helper objects
    // (name-object index, internal synchronization objects, internal
    // variables...)
    node_shmem_ = bip::managed_shared_memory(
            bip::open_or_create,
            node_address_.c_str(),
            1024 + sizeof(Node));

    // Facilitates synchronized access to shmem
    node_ = node_shmem_.find_or_construct<Node>(typeid(Node).name())();

    // Let the node know this source is attached and retrieve *this's index
    slot_index_ = node_->acquireSlot();
    bound_ = true;

    // Wait for the SINK to bind and construct the shared object
    if (node_->sink_state() != NodeState::SINK_BOUND) {
        wait();

        // Self post since all loops start with wait() and we just
        // finished our wait(). This will make the first call to
        // wait() a 'freebie'
        node_->read_barrier(slot_index_).post();
        did_wait_need_post_ = false;
    }

    // Find an existing shared object constructed by the SINK
    obj_shmem_ =
            bip::managed_shared_memory(bip::open_only, obj_address_.c_str());
    std::pair<T *,std::size_t> temp = obj_shmem_.find<T>(typeid(T).name());
    sh_object_ = temp.first;

    // Only occurs when the name of the shared object does not match typeid(T).name()
    if (sh_object_ == nullptr)
        throw std::runtime_error("A Source<T> can only connect to a node bound by a Sink<T>.");

    connected_ = true;
}


template<typename T>
inline NodeState SourceBase<T>::wait() {

#ifndef NDEBUG
    // Don't use Asserts because it does not clean shmem
    if(!bound_)
        throw std::runtime_error("Source must be bound before calling wait()");
    if (did_wait_need_post_)
        throw std::runtime_error("wait() called when post() was required.");
#endif

    boost::system_time timeout = boost::get_system_time() + msec_t(10);

    // Only wait if there is a SOURCE attached to the node
    // Wait with timed wait with period check to prevent deadlocks
    while (!node_->read_barrier(slot_index_).timed_wait(timeout)) {

        // Loops checking if wait has been released
        timeout = boost::get_system_time() + msec_t(10);

        // If the sink has left the room, we should too
        if (node_->sink_state() == NodeState::END)
            break;
    }

    did_wait_need_post_ = true;

    return node_->sink_state();
}

template<typename T>
inline void SourceBase<T>::post() {

#ifndef NDEBUG
    // Don't use Asserts because it does not clean shmem
    if(!bound_)
        throw std::runtime_error("source must be bound before calling post()");
    if(!connected_)
        throw std::runtime_error("source must be connected before calling post()");
    if (!did_wait_need_post_)
        throw std::runtime_error("post() called when wait() was required.");
#endif

    if (node_->notifySourceReadComplete(slot_index_))
        node_->write_barrier.post();

    did_wait_need_post_ = false;
}

// Specializations...

template<typename T>
class Source : public SourceBase<T> {

    using SourceBase<T>::sh_object_;
    using SourceBase<T>::connected_;

public:
    T * retrieve();
    T clone() const;
};

template<typename T>
inline T * Source<T>::retrieve() {

#ifndef NDEBUG
    // Don't use Asserts because it does not clean shmem
    if (!connected_)
        throw (std::runtime_error("Source must be connected before shared object is retrieved."));
#endif

    return sh_object_;
}

template<typename T>
inline T  Source<T>::clone() const {

#ifndef NDEBUG
    // Don't use Asserts because it does not clean shmem
    if (!connected_)
        throw (std::runtime_error("Source must be connected before shared object is cloned."));
#endif

    return *sh_object_;
}

// 1. SharedCVMat

template<>
class Source<SharedCVMat> : public SourceBase<SharedCVMat> {

public:

    struct MatParameters {
        size_t cols  {0};
        size_t rows  {0};
        size_t type  {0};
        size_t bytes {0};
    };

    void connect(const std::string &address) override;

    oat::Frame retrieve() const { return frame_; }
    oat::Frame clone() const { return frame_.clone(); }
    MatParameters parameters() const { return parameters_; }

private :
    oat::Frame frame_;
    MatParameters parameters_;

};

inline void Source<SharedCVMat>::connect(const std::string &address) {

    if (connected_ || bound_)
        throw std::runtime_error("A source can only connect a "
                                 "single time to a single node.");

    // Addresses for this block of shared memory
    node_address_ = address + "_node";
    obj_address_ = address + "_obj";

    // Define shared memory
    node_shmem_ = bip::managed_shared_memory(
            bip::open_or_create,
            node_address_.c_str(),
            1024 + sizeof(Node));

    // Facilitates synchronized access to shmem
    node_ = node_shmem_.find_or_construct<Node>(typeid(Node).name())();

    // Let the acquire a source slot and get our index
    slot_index_ = node_->acquireSlot();
    bound_ = true;

    // Wait for the SINK to bind the node and provide matrix
    // header info.
    if (node_->sink_state() != NodeState::SINK_BOUND) {

        wait();

        // Self post since all loops start with wait() and we just
        // finished our wait(). This will make the first call to
        // wait() a 'freebie'
        node_->read_barrier(slot_index_).post();
        did_wait_need_post_ = false;
    }

    // Find an existing shared object constructed by the SINK
    obj_shmem_ =
            bip::managed_shared_memory(bip::open_only, obj_address_.c_str());
    std::pair<SharedCVMat *,std::size_t> temp =
            obj_shmem_.find<SharedCVMat>(typeid(SharedCVMat).name());
    sh_object_ = temp.first;

    // Generate cv::Mat header using info in shmem segment
    frame_ = oat::Frame(sh_object_->rows(),
                        sh_object_->cols(),
                        sh_object_->type(),
                        obj_shmem_.get_address_from_handle(sh_object_->data()));

    // Save parameters so that to construct cv::Mats with
    parameters_.cols = sh_object_->cols();
    parameters_.rows = sh_object_->rows();
    parameters_.type = sh_object_->type();
    parameters_.bytes = frame_.total() * frame_.elemSize();

    connected_ = true;
}

} // namespace oat

#endif	/* OAT_SOURCE_H */
