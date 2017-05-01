//******************************************************************************
//* File:   oat buffer main.cpp
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

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/interprocess/exceptions.hpp>
#include <boost/program_options.hpp>
#include <cpptoml.h>
#include <opencv2/core.hpp>
#include <zmq.hpp>

#include "../../lib/datatypes/Pose.h"
#include "../../lib/datatypes/Frame2.h"
#include "../../lib/utility/make_unique.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/ProgramOptions.h"

#include "TokenBuffer.h"

#define REQ_POSITIONAL_ARGS 3

namespace po = boost::program_options;

using PoseBuffer = oat::TokenBuffer<oat::Pose>;
using FrameBuffer = oat::TokenBuffer<oat::SharedFrame, oat::SharedFrameAllocator>;

const char usage_type[] =
    "TYPE\n"
    "  frame: Frame buffer\n"
    "  pose: Pose buffer";

const char usage_io[] =
    "SOURCES:\n"
    "  User-supplied name of the memory segment(s) to receive tokens "
    "from (e.g. input1, input2).\n\n"
    "SINK:\n"
    "  User-supplied name of the memory segment to publish tokens "
    "to (e.g. output).";

const char purpose[] =
    "Fold tokens from SOURCES into a FIFO and resample. Publish tokens in "
    "FIFO to SINK.";

void printUsage(const po::options_description &options, const std::string &type)
{
    if (type.empty()) {
        std::cout <<
        "Usage: buffer [INFO]\n"
        "   or: buffer TYPE SOURCES SINK [CONFIGURATION]\n";

        std::cout << purpose << "\n";
        std::cout << options << "\n";
        std::cout << usage_type << "\n\n";
        std::cout << usage_io << std::endl;

    } else {
        std::cout <<
        "Usage: buffer " << type << " [INFO]\n"
        "   or: buffer " << type << " SOURCES SINK [CONFIGURATION]\n";

        std::cout << purpose << "\n\n";
        std::cout << usage_io << "\n";
        std::cout << options;
    }
}

int main(int argc, char *argv[])
{
    // Results of command line input
    std::string type;
    std::vector<std::string> addrs;

    // Component specializations
    std::unordered_map<std::string, char> type_hash;
    type_hash["frame"] = 'a';
    type_hash["pose"] = 'b';

    // The component itself
    std::string comp_name = "buffer";
    std::unique_ptr<oat::Component> buffer;

    // Program options
    po::options_description visible_options;

    try {

        // Required positional options
        po::options_description positional_opt_desc("POSITIONAL");
        positional_opt_desc.add_options()
            ("type", po::value<std::string>(&type),
             "Type of token stored by the buffer.")
            ("sources-and-sink", po::value< std::vector<std::string> >(&addrs),
            "The names the SOURCES supplying the token streams to be "
            "buffered and interleaved followed by the name of the SINK to which "
            "the combined token stream will be published.")
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

        // Collect unrecognized options. If the first positional option was
        // specific (type), delete it.
        auto special_opt =
            po::collect_unrecognized(parsed_opt.options, po::include_positional);

        if (option_map.count("type"))
            special_opt.erase(special_opt.begin(), special_opt.begin() + 1);

        // Reparse special_opt to get souces and sink.
        po::positional_options_description detail_pos_opts;
        detail_pos_opts.add("sources-and-sink", -1);

        po::store(po::command_line_parser(special_opt)
                 .options(options)
                 .positional(detail_pos_opts)
                 .run(), option_map);

        // Check options for errors and bind options to local variables
        po::notify(option_map);

        // Check IO arguments
        bool io_error {false};
        std::string io_error_msg;
        io_error = addrs.size() < (REQ_POSITIONAL_ARGS - 1);

        // If a TYPE was provided, then specialize the filter and corresponding
        // program options
        if (option_map.count("type") && !io_error) {

            // Pull out sources and sink
            std::vector<std::string> sources(addrs.begin(), addrs.end() - 1);
            auto sink = addrs.back();

            // Refine component type
            switch (type_hash[type]) {
                case 'a':
                {
                    buffer = oat::make_unique<FrameBuffer>(sources, addrs.back());
                    break;
                }
                case 'b':
                {
                    buffer = oat::make_unique<PoseBuffer>(sources, addrs.back());
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
            buffer->appendOptions(detail_opts);
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

        // Check IO arguments
        if (!option_map.count("type")) {
            io_error_msg += "A TYPE must be specified.\n";
            io_error = true;
        }

        if (!option_map.count("sources-and-sink") || addrs.size() < 2) {
            io_error_msg += "At least one SOURCE and a single SINK must be specified.\n";
            io_error = true;
        }

        if (io_error) {
            printUsage(visible_options, type);
            std::cerr << oat::Error(io_error_msg);
            return -1;
        }

        // Get specialized component name
        comp_name = buffer->name();

        // TODO: This segfaults when one source and no sink is given!!!
        // Reparse specialized component options
        special_opt =
            po::collect_unrecognized(parsed_opt.options, po::include_positional);
        special_opt.erase(special_opt.begin(),special_opt.begin() + REQ_POSITIONAL_ARGS);

        po::store(po::command_line_parser(special_opt)
                 .options(options)
                 .run(), option_map);
        po::notify(option_map);

        buffer->configure(option_map);

        std::cout << oat::whoMessage(buffer->name(), "Listening to sources ");
        for (auto s : addrs)
            std::cout << oat::sourceText(s) << " ";
        std::cout << ".\n"
                  << oat::whoMessage(buffer->name(),
                     "Steaming to sink " + oat::sinkText(addrs.back()) + ".\n")
                  << oat::whoMessage(buffer->name(),
                     "Press CTRL+C to exit.\n");

        // Infinite loop until ctrl-c or end of stream signal
        buffer->run();

        // Tell user
        std::cout << oat::whoMessage(comp_name, "Exiting.")
                  << std::endl;

        // Exit success
        return 0;

    } catch (const po::error &ex) {
        printUsage(visible_options, type);
        std::cerr << oat::whoError(comp_name, ex.what()) << std::endl;
    } catch (const cpptoml::parse_exception &ex) {
        std::cerr << oat::whoError(comp_name + "(TOML) ", ex.what()) << std::endl;
    } catch (const zmq::error_t &ex) {
        std::cerr << oat::whoError(comp_name + "(ZMQ) " , ex.what()) << std::endl;
    } catch (const cv::Exception &ex) {
        std::cerr << oat::whoError(comp_name + "(OPENCV) ", ex.what()) << std::endl;
    } catch (const boost::interprocess::interprocess_exception &ex) {
        std::cerr << oat::whoError(comp_name + "(SHMEM) ", ex.what()) << std::endl;
    } catch (const std::runtime_error &ex) {
        std::cerr << oat::whoError(comp_name, ex.what()) << std::endl;
    } catch (...) {
        std::cerr << oat::whoError(comp_name, "Unknown exception.")
                  << std::endl;
    }

    // Exit failure
    return -1;
}
