//******************************************************************************
//* File:   Buffer.h
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

#ifndef OAT_BUFFER_H
#define OAT_BUFFER_H

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include <boost/lockfree/spsc_queue.hpp>

#include "../../lib/base/Component.h"
#include "../../lib/base/Configurable.h"
#include "../../lib/shmemdf/Sink2.h"
#include "../../lib/shmemdf/Source.h"

namespace oat {

class Buffer : public Component, public Configurable {

public:
    /**
     * @brief Abstract Buffer. All concrete buffers implement this ABC.
     * @param source_address SOURCE node address
     * @param sink_address SINK node address
     */
    Buffer(const std::vector<std::string> &source_address,
           const std::string &sink_address);
    virtual ~Buffer();

    // Component Interface
    std::string name() const override { return name_; }
    ComponentType type() const override { return ComponentType::buffer; }

protected:
    static constexpr size_t BUFFSIZE{1000};
    using buffer_size_t = boost::lockfree::capacity<BUFFSIZE>;
    using msec = std::chrono::milliseconds;

    /**
     * @brief In response to downstream request, publish object from FIFO to
     * SINK.
     */
    virtual void pop(void) = 0;

    // Buffer name
    const std::string name_;

    // Sample rate ratio
    size_t down_sample_factor_{1};

    // Sink
    std::atomic<bool> sink_running_{true};
    std::thread sink_thread_;
    std::mutex cv_m_;
    std::condition_variable cv_;
    const std::string sink_address_;

private:
    // Configurable interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;
};

#ifndef NDEBUG

static constexpr size_t prog_bar_width{80};

template <typename T>
void showBufferState(const T &buffer, size_t buffer_size)
{

    std::cout << "[";

    int progress = (prog_bar_width * buffer.read_available()) / buffer_size;
    int remaining = prog_bar_width - progress;

    for (int i = 0; i < progress; ++i)
        std::cout << "=";
    for (int i = 0; i < remaining; ++i)
        std::cout << " ";

    std::cout << "] "
              + std::to_string(buffer.read_available())
              + "/"
              + std::to_string(buffer_size)
              + "\n";
}
#endif

}      /* namespace oat */
#endif /* OAT_BUFFER_H */
