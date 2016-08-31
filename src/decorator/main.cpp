//******************************************************************************
//* File:   oat decorate main.cpp
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

//NOTE: this component is a bit of a pile. I have decided not to improve upon
//its rather stupid implementation until a complete overhaul is warrented.

#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include <cpptoml.h>
#include <boost/program_options.hpp>
#include <boost/interprocess/exceptions.hpp>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/ProgramOptions.h"

#include "Decorator.h"

#define REQ_POSITIONAL_ARGS 2

namespace po = boost::program_options;

volatile sig_atomic_t quit = 0;
volatile sig_atomic_t source_eof = 0;

void run(std::shared_ptr<oat::Decorator> decorator) {

    try {

        decorator->connectToNodes();

        while (!quit && !source_eof)
            source_eof = decorator->process();

    } catch (const boost::interprocess::interprocess_exception &ex) {

        // Error code 1 indicates a SIGNINT during a call to wait(), which
        // is normal behavior
        if (ex.get_error_code() != 1)
            throw;
    }
}

void printUsage(po::options_description options) {
    std::cout << "Usage: decorate [INFO]\n"
              << "   or: decorate SOURCE SINK [CONFIGURATION]\n"
              << "Decorate the frames from SOURCE, e.g. with object position "
              << "markers and sample number. Publish decorated frames to SINK.\n\n"
              << "SOURCE:\n"
              << "  User-supplied name of the memory segment from which frames "
              << "are received (e.g. raw).\n\n"
              << "SINK:\n"
              << "  User-supplied name of the memory segment to publish frames "
              << "to (e.g. out).\n"
              << options << "\n";
}

// Signal handler to ensure shared resources are cleaned on exit due to ctrl-c
void sigHandler(int) {
    quit = 1;
}

int main(int argc, char *argv[]) {

    std::signal(SIGINT, sigHandler);

    // Results of command line input
    std::string source;
    std::string sink;

    // The component itself
    std::string comp_name = "decorator";
    std::shared_ptr<oat::Decorator> decorator;

    // Program options
    po::options_description visible_options;

    try {

        // Required positional options
        po::options_description positional_opt_desc("POSITIONAL");
        positional_opt_desc.add_options()
                ("source", po::value<std::string>(&source),
                 "User-supplied name of the memory segment to receive frames.")
                ("sink", po::value<std::string>(&sink),
                 "User-supplied name of the memory segment to publish frames.")
                ("type-args", po::value<std::vector<std::string> >(),
                 "type-specific arguments.")
                ;

        // Required positional arguments and type-specific configuration
        po::positional_options_description positional_options;
        positional_options.add("source", 1);
        positional_options.add("sink", 1);
        positional_options.add("type-args", -1);

        // Visible options for help message
        visible_options.add(oat::config::ComponentInfo::instance()->get());

        po::options_description options;
        options.add(positional_opt_desc)
               .add(oat::config::ComponentInfo::instance()->get());

        // Parse options, including unrecognized options which may be
        // type-specific
        auto parsed_opt = po::command_line_parser(argc, argv)
            .options(options)
            .positional(positional_options)
            .allow_unregistered()
            .run();

        po::variables_map option_map;
        po::store(parsed_opt, option_map);

        // Check options for errors and bind options to local variables
        po::notify(option_map);

        // Make the component (no type specialization here)
        decorator = std::make_shared<oat::Decorator>(source, sink);

        // Specialize program options for the selected TYPE
        po::options_description detail_opts {"CONFIGURATION"};
        decorator->appendOptions(detail_opts);
        visible_options.add(detail_opts);
        options.add(detail_opts);

        // Check INFO arguments
        if (option_map.count("help")) {
            printUsage(visible_options);
            return 0;
        }

        if (option_map.count("version")) {
            std::cout << oat::config::VERSION_STRING;
            return 0;
        }

        // Check IO arguments
        bool io_error {false};
        std::string io_error_msg;

        if (!option_map.count("source")) {
            io_error_msg += "A SOURCE must be specified.\n";
            io_error = true;
        }

        if (!option_map.count("sink")) {
            io_error_msg += "A SINK must be specified.\n";
            io_error = true;
        }

        if (io_error) {
            printUsage(visible_options);
            std::cerr << oat::Error(io_error_msg);
            return -1;
        }

        // Get specialized component name
        comp_name = decorator->name();

        // Reparse specialized component options
        auto special_opt =
            po::collect_unrecognized(parsed_opt.options, po::include_positional);
        special_opt.erase(special_opt.begin(),special_opt.begin() + REQ_POSITIONAL_ARGS);

        po::store(po::command_line_parser(special_opt)
                 .options(options)
                 .run(), option_map);
        po::notify(option_map);

        decorator->configure(option_map);

        // Tell user
        std::cout << oat::whoMessage(decorator->name(),
            "Listening to source " + oat::sourceText(source) + ".\n")
            << oat::whoMessage(decorator->name(),
            "Steaming to sink " + oat::sinkText(sink) + ".\n")
            << oat::whoMessage(decorator->name(),
            "Press CTRL+C to exit.\n");

        // Infinite loop until ctrl-c or end of stream signal
        run(decorator);

        // Tell user
        std::cout << oat::whoMessage(comp_name, "Exiting.\n");

        // Exit
        return 0;

    } catch (const po::error &ex) {
        printUsage(visible_options);
        std::cerr << oat::whoError(comp_name, ex.what()) << std::endl;
    } catch (const cpptoml::parse_exception &ex) {
        std::cerr << oat::whoError(comp_name,"Invalid TOML syntax\n")
                  << oat::whoError(comp_name, ex.what())
                  << std::endl;
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

    // Exit failure
    return -1;
}

