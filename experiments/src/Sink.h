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

#ifndef SINK_H
#define	SINK_H

#include <string>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/thread/thread_time.hpp>

#include "Node.h"

namespace oat {
    
    namespace bip = boost::interprocess;
    using handle_t = bip::managed_shared_memory::handle_t;
    using msec = boost::posix_time::milliseconds;
    //using ipvector = boost::interprocess::vector<std::shared_ptr<semaphore>>;
    class Node;

template<typename T>
class SinkBase {
public:
    SinkBase();
    virtual ~SinkBase();
    
    // Sinks cannot be copied
    SinkBase &operator=(const SinkBase &) = delete;
    SinkBase(const SinkBase& orig) = delete;
    
    void bind(const std::string &address, const size_t bytes);

    void wait();
    void post();

protected:
    
    Node* node_;
    T* object_;
    bip::managed_shared_memory shmem_;
        
private:
    
    std::string shmem_address_;
};

template<typename T>
SinkBase<T>::SinkBase()  
{
    // Nothing
}

template<typename T>
SinkBase<T>::~SinkBase() {

    // Detach this server from shared mat header
    node_->set_sink_state(oat::SinkState::END);

    // If the client ref count is 0, memory can be deallocated
    if (node_->source_ref_count() == 0) {

        // Remove_shared_memory on object destruction
        bip::shared_memory_object::remove(shmem_address_.c_str());
//#ifndef NDEBUG
//        std::cout << oat::dbgMessage("Shared memory \'" + shmem_address_ + "\' was deallocated.\n");
//#endif
    }
}

template<typename T>
void SinkBase<T>::bind(const std::string& address, const size_t bytes) {

    // Addresses for this block of shared memory
    shmem_address_ = address;
    std::string node_address = shmem_address_ + "/shmgr";
    std::string obj_address = shmem_address_ + "/shobj";
    std::string shrb_address =  shmem_address_ + "/shrb";

    // Define shared memory
    shmem_ = bip::managed_shared_memory(
            bip::open_or_create,
            address.c_str(),
            bytes + sizeof (oat::Node) + sizeof(T));

    // Facilitates synchronized access to shmem
    node_ = shmem_.find_or_construct<oat::Node>(node_address.c_str())();

    // Make sure there is not another SINK using this shmem
    if (node_->sink_state() != oat::SinkState::UNDEFINED) {

        // There is already a SINK using this shmem
        throw (std::runtime_error(
                "Requested SINK address, '" + shmem_address_ + "', is not available."));

    } else {

        // Find an existing shared object or construct one with default parameters
        object_ = shmem_.find_or_construct<T>(obj_address.c_str())();
        node_->getReadBarrier(shrb_address.c_str(), shmem_);
        //shmem_bound_ = true;
        node_->set_sink_state(oat::SinkState::BOUND);
    }
}

template<typename T>
void SinkBase<T>::wait() {
    
    boost::system_time timeout = boost::get_system_time() + msec(10);
    
    // Only wait if there is a SOURCE attached to the node
    // Wait with timed wait with period check to prevent deadlocks
    while (node_->source_ref_count() > 0 && 
           !node_->write_barrier.timed_wait(timeout)) {
        // Loops checking if wait has been released
        timeout = boost::get_system_time() + msec(10);
    }

    // Obtain ownership over mutex so that shared memory can be written to. 
    // This must be released using signal()
    //manager_->mutex.wait();
}

template<typename T>
void SinkBase<T>::post() {
    
    // Release exclusive lock over shared memory
    //manager_->mutex.post();
    node_->incrementWriteNumber();
    
    // Reset the source read count
    node_->resetSourceReadCount();
    
    // Tell each source they can proceed
    for (shared_ptr_vector_t::size_type i = 0; i < node_->source_ref_count(); i++) 
            
            //auto &s : node_->read_barrier) {
        //s->post(); 
        node_->read_barrier->at(i)->post();
    //}
}

// Specializations
template<typename T>
class Sink : public SinkBase<T> { };

template<>
class Sink<SharedCVMat> : public SinkBase<SharedCVMat> { 

    public:
        cv::Mat allocate(const cv::Size dims, const int type); 
        
};

cv::Mat Sink<SharedCVMat>::allocate(const cv::Size dims, const int type) {

    // TODO: Would be best to grow the segment to the right size here, but in
    // order to do this, no process can be mapped to the segment. This
    // will be very hard to achieve since member objects (e.g. manager_) are
    // mapped and hang around from the lifetime of *this
    //bip::managed_shared_memory::grow(shmem_address_.c_str(), data_size);
    
    // Make sure that the SINK is bound to a shared memory segment
    if (node_->sink_state() != oat::SinkState::BOUND)
        throw (std::runtime_error("SINK must be bound before memory is allocated."));

    // Allocate memory for the shared object's data
    cv::Mat temp(dims, type); 
    void *data = shmem_.allocate(temp.total() * temp.elemSize());
    handle_t handle = shmem_.get_handle_from_address(data);
    
    // Reset the SharedCVMat's parameters now that we know what they should be
    object_->setParameters(handle, dims, type);
    
    // Return pointer to memory allocated for shared object
    return cv::Mat(dims, type, data);
}

} // namespace oat

#endif	/* SINK_H */

