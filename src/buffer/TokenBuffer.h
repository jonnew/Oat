//******************************************************************************
//* File:   TokenBuffer.h
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

#ifndef OAT_TOKEN_BUFFER_H
#define	OAT_TOKEN_BUFFER_H

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include <boost/lockfree/spsc_queue.hpp>

#include "../../lib/base/Component.h"
#include "../../lib/shmemdf/Helpers.h"
#include "../../lib/shmemdf/Sink2.h"
#include "../../lib/shmemdf/Source.h"
#include "../../lib/utility/IOFormat.h"

namespace oat {

/**
 * Generic token buffer.
 */
template <typename T, typename RT = T, typename A = int>
class TokenBuffer : public Component {

    static constexpr size_t buffer_size{1000};
    using BufferSizeT = boost::lockfree::capacity<buffer_size>;
    using MSec = std::chrono::milliseconds;
    using SPSCBuffer = boost::lockfree::spsc_queue<T, BufferSizeT>;
    using SourceT = oat::Source<T>;

public:
    TokenBuffer(const std::vector<std::string> &source_addrs,
                const std::string &sink_addr)
    : Component()
    , sink_(sink_addr)
    {
        for (auto &a : source_addrs)
            sources_.push_back(oat::make_unique<SourceT>(a));

        // Increase the sample rate in accordance with the number of sources
        // being combined
        upsample_factor_ = source_addrs.size();

        // Set the component name for this instance
        set_name(source_addrs, sink_addr);
    }

    ~TokenBuffer()
    {
        // Join threads
        sink_running_ = false;
        if (sink_thread_.joinable())
            sink_thread_.join();
    }

    // Component Interface
    ComponentType type() const override { return ComponentType::buffer; }

private:
    // Buffer name
    //const std::string name_;

    // Configurable interface
    po::options_description options() const override {

        // Update CLI options
        po::options_description local_opts;
        local_opts.add_options()
            ("down-sample-factor,d", po::value<size_t>(),
             "Positive integer value, specifying the token decimation factor of "
             "the buffer. The outgoing stream's sample rate will be the "
             "incoming streams' rate divided by this number times the number of "
             "streams being folded.  Defaults to 1.")
            ;

        return local_opts;
    }

    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override
    {
        // Resample ratio
        oat::config::getNumericValue<size_t>(
            vm, config_table, "down-sample-factor", downsample_factor_, 1);
    }

    bool connectToNode(void) override {

        // Wait for synchronous start with sink when it binds its node
        bool connected = true;
        std::vector<double> periods;
        for (auto &s : sources_) {
            connected |= (s->connect() == SourceState::connected);
            periods.push_back(s->retrieve()->period().count());
        }

        double min_rate;
        if (!checkSamplePeriods(periods, min_rate)) {
            throw std::runtime_error(
                "Folded buffer sources must have the same sample period.");
        }

        // Bind sink
        sink_.bind(oat::Token::Seconds(1.0 / min_rate));

        // Start buffer pop thread
        sink_thread_ = std::thread(&TokenBuffer<T>::pop, this);

        return connected;
    }

    int process(void) override
    {
        std::unique_ptr<T> token;

        // Synchronous pull from all sources
        int rc = 0;
        for (size_t i = 0; i < sources_.size(); i++) {

            rc = sources_[i]->pull(token);
            if (rc) { return rc; }

            if (token->tick() % downsample_factor_ == 0) {

                token->resample(upsample_factor_, downsample_factor_, i);

                if (!buffer_.push(*token))
                    std::cerr << oat::Error("Buffer overrun.\n");
            }
        }

        // Notify comsumer thread that it should proceed
        cv_.notify_one();

#ifndef NDEBUG
        showBufferState(buffer_, buffer_size);
#endif

        // Sources were not at EOF state
        return rc;
    }

    /**
     * @brief In response to downstream request, publish object from FIFO to
     * SINK.
     */
    void pop(void)
    {
        while (sink_running_) { // Prevent infinite loop when sink exits

            // Proceed only if buffer_ has data
            std::unique_lock<std::mutex> lk(cv_m_);
            if (cv_.wait_for(lk, MSec(10)) == std::cv_status::timeout)
                continue;

            // Publish objects when they are requested until the buffer
            // is empty
            buffer_.consume_all([this](T &t) { sink_.push(std::move(t)); });
        }
    }

    // Token pop thread
    std::atomic<bool> sink_running_{true};
    std::thread sink_thread_;
    std::mutex cv_m_;
    std::condition_variable cv_;

    // Source
    std::vector<std::unique_ptr<SourceT>> sources_;

    // Sample rate ratio
    size_t upsample_factor_{1};
    size_t downsample_factor_{1};

    // Buffer
    SPSCBuffer buffer_;

    // Sink
    oat::SinkBase<T, RT, A> sink_;
};

template <>
bool TokenBuffer<oat::SharedFrame, oat::Frame, SharedFrameAllocator>::connectToNode(void)
{
    // Wait for synchronous start with sink when it binds its node
    bool connected = true;
    std::vector<double> periods;
    oat::SharedFrame *first;
    for (size_t i = 0; i < sources_.size(); i++) {
        connected |= (sources_[i]->connect() == SourceState::connected);

        // Check that frames have equal parameters
        if (i == 0) {
            first = sources_[i]->retrieve();
        } else {
            if (!oat::SharedFrame::compare(*first, *sources_[i]->retrieve())) {
                throw std::runtime_error(
                    "Folded frame sources must provide the same frame types.");
            }
        }

        periods.push_back(sources_[i]->retrieve()->period().count());
    }

    double min_rate;
    if (!checkSamplePeriods(periods, min_rate)) {
        throw std::runtime_error(
            "Folded buffer sources must have the same sample period.");
    }

    // Bind sink
    sink_.reserve(first->bytes());
    sink_.bind(oat::Token::Seconds(1.0 / min_rate),
               first->rows(),
               first->cols(),
               first->color());

    // Start buffer pop thread
    sink_thread_ = std::thread(
        &TokenBuffer<SharedFrame, SharedFrameAllocator>::pop, this);

    return connected;
}


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
#endif /* OAT_TOKEN_BUFFER_H */
