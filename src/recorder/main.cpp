//******************************************************************************
//* File:   oat record main.cpp
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
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
//****************************************************************************

#include <algorithm>
#include <csignal>
#include <thread>
#include <pthread.h> // TODO: POSIX specific
//#include <functional>
#include <memory>
#include <unordered_map>
#include <string>
#include <zmq.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/program_options.hpp>

#include "../../lib/utility/ZMQStream.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/IOUtility.h"
#include "../../lib/utility/ProgramOptions.h"

#include "RecordControl.h"
#include "Recorder.h"

namespace po = boost::program_options;

volatile sig_atomic_t quit = 0;
volatile sig_atomic_t source_eof = 0;
bool first_loop = true;

// Needed by both threads
std::string file_name;
std::string save_path;
bool allow_overwrite = false;
bool prepend_timestamp = false;
bool concise_file = false;

// ZMQ stream
using zmq_istream_t = boost::iostreams::stream<oat::zmq_istream>;
using zmq_ostream_t = boost::iostreams::stream<oat::zmq_ostream>;

// Interactive commands
bool recording_on = true;

void printUsage(std::ostream &out, po::options_description options)
{
    out << "Usage: record [INFO]\n"
        << "   or: record [CONFIGURATION]\n"
        << "Record any Oat token source(s).\n"
        << options << "\n";
}

// Signal handler to ensure shared resources are cleaned on exit due to ctrl-c
void sigHandler(int)
{
    quit = 1;
}

// Cleanup procedure for interactive sessions
void cleanup(std::thread &proc_thread) {

    // Reinstall SIGINT handler and trigger it on both threads
    std::signal(SIGINT, sigHandler);
    pthread_kill(proc_thread.native_handle(), SIGINT);

    // Join recorder and UI threads
    proc_thread.join();
}

// Processing loop
void run(std::shared_ptr<oat::Recorder> &recorder)
{
    try {

        recorder->connectToNodes();

        while (!quit && !source_eof)
            source_eof = recorder->writeStreams();

    } catch (const boost::interprocess::interprocess_exception &ex) {

        // Error code 1 indicates a SIGINT during a call to wait(), which
        // is normal behavior
        if (ex.get_error_code() != 1)
            throw;
    }
}

int main(int argc, char *argv[]) {

    std::signal(SIGINT, sigHandler);

    // The component itself
    std::string comp_name = "recorder";

    // Program options
    po::options_description options;

    try {

        int rc = 1;
        while (rc == 1) {


            // We may be coming around for another recording so reset quit
            // flag and remove all options
            quit = 0;

            auto recorder = std::make_shared<oat::Recorder>();

            // Config options
            if (first_loop) {
                options.add(oat::config::ComponentInfo::instance()->get());
                po::options_description detail_opts {"CONFIGURATION"};
                recorder->appendOptions(detail_opts);
                options.add(detail_opts);
            }

            // Parse options, including unrecognized options which may be
            // type-specific
            auto parsed_opt = po::command_line_parser(argc, argv)
                .options(options)
                .run();

            po::variables_map option_map;
            po::store(parsed_opt, option_map);

            // Check options for errors and bind options to local variables
            po::notify(option_map);

            // Check INFO arguments
            if (option_map.count("help")) {
                printUsage(std::cout, options);
                return 0;
            }

            if (option_map.count("version")) {
                std::cout << oat::config::VERSION_STRING;
                return 0;
            }

            // Configure recorder parameters
            recorder->configure(option_map);

            // Ger real name
            comp_name = recorder->name();


            std::cout << oat::whoMessage(recorder->name(),
                    "Press CTRL+C to exit.\n");

            switch (recorder->control_mode)
            {
                case oat::Recorder::ControlMode::NONE :
                {
                    // Start the recorder w/o controls
                    rc = 0;
                    run(recorder);
                    break;
                }
                case oat::Recorder::ControlMode::LOCAL :
                {
                    // For interactive control, recorder must be started by user
                    recorder->set_record_on(false);

                    // Start recording in background
                    std::thread process(run, std::ref(recorder));
                    try {
                      // Interact using stdin
                      oat::printInteractiveUsage(std::cout);
                      rc = oat::controlRecorder(std::cin, std::cout, *recorder, true);
                    } catch (...) {
                      // Interrupt and join threads
                      cleanup(process);
                      throw;
                    }
                    cleanup(process);

                    break;
                }
                case oat::Recorder::ControlMode::RPC :
                {
                    // For interactive control, recorder must be started by user
                    recorder->set_record_on(false);

                    // Start recording in background
                    std::thread process(run, std::ref(recorder));

                    try {
                        auto ctx = std::make_shared<zmq::context_t>(1);
                        auto sock = std::make_shared<zmq::socket_t>(*ctx, ZMQ_REP);
                        sock->bind(recorder->rpc_endpoint().c_str());
                        zmq_istream_t in(ctx, sock);
                        zmq_ostream_t out(ctx, sock);
                        oat::printRemoteUsage(std::cout);
                        rc = oat::controlRecorder(in, out, *recorder, false);
                    } catch (const zmq::error_t &ex) {
                        std::cerr << oat::whoError(recorder->name(), "zeromq error: "
                                + std::string(ex.what())) << "\n";
                        cleanup(process);
                        return -1;
                    }

                    // Interupt and join threads
                    cleanup(process);

                    break;
                }
            }

            first_loop = false;
        }

        // Exit
        std::cout << oat::whoMessage(comp_name, "Exiting.\n");
        return 0;

    } catch (const po::error &ex) {
        printUsage(std::cout, options);
        std::cerr << oat::whoError(comp_name, ex.what()) << std::endl;
    } catch (const cpptoml::parse_exception &ex) {
        std::cerr << oat::whoError(comp_name, ex.what()) << std::endl;
    } catch (const std::runtime_error &ex) {
        std::cerr << oat::whoError(comp_name,ex.what()) << std::endl;
    } catch (const cv::Exception &ex) {
        std::cerr << oat::whoError(comp_name, ex.what()) << std::endl;
    } catch (const boost::interprocess::interprocess_exception &ex) {
        std::cerr << oat::whoError(comp_name, ex.what()) << std::endl;
    } catch (...) {
        std::cerr << oat::whoError(comp_name, "Unknown exception.")
                  << std::endl;
    }

    // exit failure
    return -1;
}
