//******************************************************************************
//* File:   oat posicom main.cpp
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

#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cpptoml.h>
#include <boost/program_options.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <opencv2/core.hpp>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/ProgramOptions.h"

#include "PositionCombiner.h"
#include "MeanPosition.h"

#define REQ_POSITIONAL_ARGS 1

namespace po = boost::program_options;

volatile sig_atomic_t quit = 0;
volatile sig_atomic_t source_eof = 0;

const char usage_type[] =
    "TYPE\n"
    "  mean: Geometric mean of positions";

const char usage_io[] =
    "SOURCES:\n"
    "  User-supplied position source names (e.g. pos1 pos2).\n\n"
    "SINK:\n"
    "  User-supplied position sink name (e.g. pos).";

const char purpose[] =
    "Combine positional information from two or more SOURCES and "
    "Publish combined position to SINK.";

void printUsage(const po::options_description &options,
                const std::string &type) {

    if (type.empty()) {
        std::cout <<
        "Usage: posicom [INFO]\n"
        "   or: posicom TYPE SOURCES SINK [CONFIGURATION]\n";

        std::cout << purpose << "\n";
        std::cout << options << "\n";
        std::cout << usage_type << "\n\n";
        std::cout << usage_io << std::endl;

    } else {
        std::cout <<
        "Usage: posicom " << type << " [INFO]\n"
        "   or: posicom " << type << " SOURCES SINK [CONFIGURATION]\n";

        std::cout << purpose << "\n\n";
        std::cout << usage_io << "\n";
        std::cout << options;
    }
}

// Signal handler to ensure shared resources are cleaned on exit due to ctrl-c
void sigHandler(int) {
    quit = 1;
}

// Processing loop
void run(const std::shared_ptr<oat::PositionCombiner>& combiner) {

    try {

        combiner->connectToNodes();

        while (!quit && !source_eof)
            source_eof = combiner->process();

    } catch (const boost::interprocess::interprocess_exception &ex) {

        // Error code 1 indicates a SIGNINT during a call to wait(), which
        // is normal behavior
        if (ex.get_error_code() != 1)
            throw;
    }
}

int main(int argc, char *argv[]) {

    std::signal(SIGINT, sigHandler);

    // Results of command line input
    std::vector<std::string> sources;
    std::string sink;
    std::string type;

    // Component specializations
    std::unordered_map<std::string, char> type_hash;
    type_hash["mean"] = 'a';

    // The component itself
    std::string comp_name = "posicom";
    std::shared_ptr<oat::PositionCombiner> combiner;

    // program options
    po::options_description visible_options;

    try {

        // Required positional options
        po::options_description positional_opt_desc("POSITIONAL");
        positional_opt_desc.add_options()
            ("type", po::value<std::string>(&type),
            "Type of test position combiner to use.")
            ("sources-and-sink", po::value< std::vector<std::string> >(),
            "The names the SOURCES supplying the positions streams to be "
            "combined followed by the name of the SINK to which the combined "
            "position will be published.")
            ("type-args", po::value<std::vector<std::string> >(),
             "type-specifuc arguments.")
            ;

        // Required positional arguments and type-specific configuration
        po::positional_options_description positional_options;
        positional_options.add("type", 1);
        positional_options.add("type-args", -1);

        // Visible options for help message
        visible_options.add(oat::config::ComponentInfo::instance()->get());

        // All options, including positional
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

        // If a TYPE was provided, then specialize
        if (option_map.count("type")) {
            switch (type_hash[type]) {
                case 'a':
                {
                    combiner = std::make_shared<oat::MeanPosition>();
                    break;
                }
                default:
                {
                    printUsage(visible_options, "");
                    std::cerr << oat::Error("Invalid TYPE specified.\n");
                    return -1;
                }
            }

            // Specialize program options for the selected TYPE
            po::options_description detail_opts {"CONFIGURATION"};
            combiner->appendOptions(detail_opts);
            visible_options.add(detail_opts);
            options.add(detail_opts);
        }

        // Check INFO arguments
        if (option_map.count("help")) {
            printUsage(visible_options, type);
            return 0;
        }

        if (option_map.count("version")) {
            std::cout << oat::config::VERSION_STRING;
            return 0;
        }

        // Get specialized component name
        comp_name = combiner->name();

        // Check IO arguments
        bool io_error {false};
        std::string io_error_msg;
        if (!option_map.count("type")) {
            io_error_msg += "A TYPE must be specified.\n";
            io_error = true;
        }

        // TODO: Ugly
        // Read unlimited positional options to get sources and sink
        po::positional_options_description detail_pos_opts;
        detail_pos_opts.add("sources-and-sink", -1);

        // Reparse specialized component options
        auto special_opt =
            po::collect_unrecognized(parsed_opt.options, po::include_positional);
        special_opt.erase(special_opt.begin(),special_opt.begin() + REQ_POSITIONAL_ARGS);

        po::store(po::command_line_parser(special_opt)
                 .options(options)
                 .positional(detail_pos_opts)
                 .run(), option_map);

        po::notify(option_map);

        if (!option_map.count("sources-and-sink")) {
            io_error_msg += "At least twos SOURCEs and one SINK must be specified.\n";
            io_error = true;
        }

        if (io_error) {
            printUsage(visible_options, type);
            std::cerr << oat::Error(io_error_msg);
            return -1;
        }

        combiner->configure(option_map);

        std::cout << oat::whoMessage(combiner->name(), "Listening to sources ");
        for (auto s : sources) { std::cout << oat::sourceText(s) << " "; }
        std::cout << ".\n"
                  << oat::whoMessage(combiner->name(),
                     "Steaming to sink " + oat::sinkText(sink) + ".\n")
                  << oat::whoMessage(combiner->name(),
                     "Press CTRL+C to exit.\n");

        // Infinite loop until ctrl-c or server end-of-stream signal
        run(combiner);

        // Tell user
        std::cout << oat::whoMessage(comp_name, "Exiting.")
                  << std::endl;

        // Exit success
        return 0;

    } catch (const po::error &ex) {
        printUsage(visible_options, type);
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

    // Exit failure
    return -1;
}
