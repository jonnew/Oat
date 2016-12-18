//******************************************************************************
//* File:   oat framefilt main.cpp
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

#include <boost/interprocess/exceptions.hpp>
#include <boost/program_options.hpp>
#include <cpptoml.h>
#include <opencv2/core.hpp>
#include <zmq.hpp>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/ProgramOptions.h"

#include "BackgroundSubtractor.h"
#include "BackgroundSubtractorMOG.h"
#include "ColorConvert.h"
#include "FrameFilter.h"
#include "FrameMasker.h"
#include "Undistorter.h"
#include "Threshold.h"

#define REQ_POSITIONAL_ARGS 3

namespace po = boost::program_options;

const char usage_type[] =
    "TYPE\n"
    "  bsub: Background subtraction\n"
    "  col: Color conversion\n"
    "  mask: Binary mask\n"
    "  mog: Mixture of Gaussians background segmentation.\n"
    "  undistort: Correct for lens distortion using lens distortion model.\n"
    "  thresh: Simple intensity threshold.";

const char usage_io[] =
    "SOURCE:\n"
    "  User-supplied name of the memory segment to receive frames "
    "from (e.g. raw).\n\n"
    "SINK:\n"
    "  User-supplied name of the memory segment to publish frames "
    "to (e.g. filt).";

const char purpose[] =
    "Filter frames from SOURCE and publish filtered frames "
    "to SINK.";

void printUsage(const po::options_description &options,
                const std::string &type) {

    if (type.empty()) {
        std::cout <<
        "Usage: framefilt [INFO]\n"
        "   or: framefilt TYPE SOURCE SINK [CONFIGURATION]\n";

        std::cout << purpose << "\n";
        std::cout << options << "\n";
        std::cout << usage_type << "\n\n";
        std::cout << usage_io << std::endl;

    } else {
        std::cout <<
        "Usage: framefilt " << type << " [INFO]\n"
        "   or: framefilt " << type << " SOURCE SINK [CONFIGURATION]\n";

        std::cout << purpose << "\n\n";
        std::cout << usage_io << "\n";
        std::cout << options;
    }
}

int main(int argc, char *argv[]) {

    // Results of command line input
    std::string type;
    std::string source;
    std::string sink;

    // Component specializations
    std::unordered_map<std::string, char> type_hash;
    type_hash["bsub"] = 'a';
    type_hash["mask"] = 'b';
    type_hash["mog"] = 'c';
    type_hash["undistort"] = 'd';
    type_hash["col"] = 'e';
    type_hash["thresh"] = 'f';

    // The component itself
    std::string comp_name = "framefilt";
    std::shared_ptr<oat::FrameFilter> filter;

    // Program options
    po::options_description visible_options;

    try {

        // Required positional options
        po::options_description positional_opt_desc("POSITIONAL");
        positional_opt_desc.add_options()
            ("type", po::value<std::string>(&type),
             "Type of frame filter to use.")
            ("source", po::value<std::string>(&source),
             "User-supplied name of the memory segment to receive frames.")
            ("sink", po::value<std::string>(&sink),
             "User-supplied name of the memory segment to publish frames.")
            ("type-args", po::value<std::vector<std::string> >(),
             "type-specific arguments.")
            ;

        // Required positional arguments and type-specific configuration
        po::positional_options_description positional_options;
        positional_options.add("type", 1);
        positional_options.add("source", 1);
        positional_options.add("sink", 1);
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

        // If a TYPE was provided, then specialize the component and
        // corresponding program options
        if (option_map.count("type")) {

            // Refine component type
            switch (type_hash[type]) {
                case 'a':
                {
                    filter = std::make_shared<oat::BackgroundSubtractor>(source, sink);
                    break;
                }
                case 'b':
                {
                    filter = std::make_shared<oat::FrameMasker>(source, sink);
                    break;
                }
                case 'c':
                {
                    filter = std::make_shared<oat::BackgroundSubtractorMOG>(source, sink);
                    break;
                }
                case 'd':
                {
                    filter = std::make_shared<oat::Undistorter>(source, sink);
                    break;
                }
                case 'e':
                {
                    filter = std::make_shared<oat::ColorConvert>(source, sink);
                    break;
                }
                case 'f':
                {
                    filter = std::make_shared<oat::Threshold>(source, sink);
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
            filter->appendOptions(detail_opts);
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
        bool io_error {false};
        std::string io_error_msg;

        if (!option_map.count("type")) {
            io_error_msg += "A TYPE must be specified.\n";
            io_error = true;
        }

        if (!option_map.count("source")) {
            io_error_msg += "A SOURCE must be specified.\n";
            io_error = true;
        }

        if (!option_map.count("sink")) {
            io_error_msg += "A SINK must be specified.\n";
            io_error = true;
        }

        if (io_error) {
            printUsage(visible_options, type);
            std::cerr << oat::Error(io_error_msg);
            return -1;
        }

        // Get specialized component name
        comp_name = filter->name();

        // Reparse specialized component options
        auto special_opt =
            po::collect_unrecognized(parsed_opt.options, po::include_positional);
        special_opt.erase(special_opt.begin(),special_opt.begin() + REQ_POSITIONAL_ARGS);

        po::store(po::command_line_parser(special_opt)
                 .options(options)
                 .run(), option_map);
        po::notify(option_map);

        filter->configure(option_map);

        // Tell user
        std::cout << oat::whoMessage(comp_name,
                     "Listening to source " + oat::sourceText(source) + ".\n")
                  << oat::whoMessage(comp_name,
                     "Steaming to sink " + oat::sinkText(sink) + ".\n")
                  << oat::whoMessage(comp_name,
                      "Press CTRL+C to exit.\n");

        // Infinite loop until ctrl-c or end of messages signal
        filter->run();

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

    // exit failure
    return -1;
}
