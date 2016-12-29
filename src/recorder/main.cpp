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

#include <memory>
#include <unordered_map>
#include <string>

#include <boost/program_options.hpp>
#include <zmq.hpp>

#include "../../lib/utility/ZMQStream.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/IOUtility.h"
#include "../../lib/utility/ProgramOptions.h"

#include "Recorder.h"

namespace po = boost::program_options;

void printUsage(const po::options_description &options)
{
    std::cout << "Usage: record [INFO]\n"
              << "   or: record [CONFIGURATION]\n"
              << "Record any Oat token source(s).\n"
              << options << "\n";
}

int main(int argc, char *argv[])
{
    // The component itself
    std::string comp_name = "recorder";
    auto recorder = std::make_shared<oat::Recorder>();

    // Program options
    po::options_description options;

    try {

        // Help and version
        options.add(oat::config::ComponentInfo::instance()->get());

        // Others
        po::options_description detail_opts{"CONFIGURATION"};
        recorder->appendOptions(detail_opts);
        options.add(detail_opts);

        // Parse options, including unrecognized options which may be
        // writer-specific
        auto parsed_opt = po::command_line_parser(argc, argv)
            .options(options)
            .allow_unregistered()
            .run();

        po::variables_map option_map;
        po::store(parsed_opt, option_map);

        // Check options for errors and bind options to local variables
        po::notify(option_map);

        // Check INFO arguments
        if (option_map.count("help")) {
            printUsage(options);
            return 0;
        }

        if (option_map.count("version")) {
            std::cout << oat::config::VERSION_STRING;
            return 0;
        }

        // Configure recorder parameters
        recorder->configure(option_map);

        // Get real name
        comp_name = recorder->name();

        std::cout
            << oat::whoMessage(recorder->name(), "Press CTRL+C to exit.\n");

        // Processing and control loops
        recorder->run();

        // Exit
        std::cout << oat::whoMessage(comp_name, "Exiting.\n");
        return 0;

    } catch (const po::error &ex) {
        printUsage(options);
        std::cerr << oat::whoError(comp_name, ex.what()) << std::endl;
    } catch (const cpptoml::parse_exception &ex) {
        std::cerr << oat::whoError(comp_name + "(TOML) ", ex.what()) << std::endl;
    } catch (const cv::Exception &ex) {
        std::cerr << oat::whoError(comp_name + "(OPENCV) ", ex.what()) << std::endl;
    } catch (const boost::interprocess::interprocess_exception &ex) {
        std::cerr << oat::whoError(comp_name + "(SHMEM) ", ex.what()) << std::endl;
    } catch (const zmq::error_t &ex) {
        if (ex.num() != EINTR)
            std::cerr << oat::whoError(comp_name + "(ZMQ) " , ex.what()) << std::endl;
    } catch (const std::runtime_error &ex) {
        std::cerr << oat::whoError(comp_name, ex.what()) << std::endl;
    } catch (...) {
        std::cerr << oat::whoError(comp_name, "Unknown exception.")
                  << std::endl;
    }

    // Exit failure
    return -1;
}
