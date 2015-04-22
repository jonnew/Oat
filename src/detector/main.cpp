//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
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

#include "Detector.h"
#include "HSVDetector.h"
#include "DifferenceDetector.h"

#include <signal.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>
#include <unordered_map>

namespace po = boost::program_options;

volatile sig_atomic_t done = 0;
bool running = true;

void term(int) {
    done = 1;
}

void printUsage(po::options_description options) {
    std::cout << "Usage: detector [OPTIONS]\n";
    std::cout << "   or: detector TYPE SOURCE SINK [CONFIGURATION]\n";
    std::cout << "Perform TYPE object detection on images from SMServer<SharedCVMatHeader> SOURCE.\n";
    std::cout << "Publish detected object positions to a SMSserver<Position2D> SINK.\n\n";
    std::cout << "TYPE\n";
    std::cout << "  diff: Difference detector (grey-scale)\n";
    std::cout << "  hsv: HSV detector (color)\n\n";
    std::cout << options << "\n";
}

// Processing thread
void run(Detector* detector) {

    while (!done) {
        detector->findObject();
        detector->servePosition();
    }
    
    detector->stop();
}

// IO thread
int main(int argc, char *argv[]) {

    signal(SIGINT, term);

    // Base options
    po::options_description options("OPTIONS");

    std::string source;
    std::string sink;
    std::string type;
    std::string config_file;
    std::string config_key;
    bool config_used = false;
    
    std::unordered_map<std::string, char> type_hash;
    type_hash["diff"] = 'a';
    type_hash["hsv"] = 'b';

    try {

        options.add_options()
                ("help", "Produce help message.")
                ("version,v", "Print version information.")
                ;

        po::options_description config("CONFIGURATION");
        config.add_options()
                ("config-file,c", po::value<std::string>(&config_file), "Configuration file.")
                ("config-key,k", po::value<std::string>(&config_key), "Configuration key.")
                ;

        po::options_description hidden("HIDDEN OPTIONS");
        hidden.add_options()
                ("type,t", po::value<std::string>(&type), "Detector type.\n\n"
                "Values:\n"
                "  diff: Difference detector (grey-scale)\n"
                "  hsv: HSV detector (color)")
                ("source", po::value<std::string>(&source),
                "The name of the SOURCE that supplies images on which hsv-filter object detection will be performed."
                "The server must be of type SMServer<SharedCVMatHeader>\n")
                ("sink", po::value<std::string>(&sink),
                "The name of the SINK to which position background subtracted images will be served.")
                ;

        po::positional_options_description positional_options;
        positional_options.add("type", 1);
        positional_options.add("source", 1);
        positional_options.add("sink", 1);

        po::options_description visible_options("VISIBLE OPTIONS");
        visible_options.add(options).add(config);

        po::options_description all_options("ALL OPTIONS");
        all_options.add(options).add(config).add(hidden);

        po::variables_map variable_map;
        po::store(po::command_line_parser(argc, argv)
                .options(all_options)
                .positional(positional_options)
                .run(),
                variable_map);
        po::notify(variable_map);

        // Use the parsed options
        if (variable_map.count("help")) {
            printUsage(visible_options);
            return 0;
        }

        if (variable_map.count("version")) {
            std::cout << "Simple-Tracker HSV Threshold Object Detector version 1.0\n"; //TODO: Cmake managed versioning
            std::cout << "Written by Jonathan P. Newman in the MWL@MIT.\n";
            std::cout << "Licensed under the GPL3.0.\n";
            return 0;
        }

        if (!variable_map.count("type")) {
            printUsage(options);
            std::cout << "Error: a TYPE must be specified. Exiting.\n";
            return -1;
        }

        if (!variable_map.count("source")) {
            printUsage(options);
            std::cout << "Error: a SOURCE must be specified. Exiting.\n";
            return -1;
        }

        if (!variable_map.count("sink")) {
            printUsage(options);
            std::cout << "Error: a SINK name must be specified. Exiting.\n";
            return -1;
        }

        if ((variable_map.count("config-file") && !variable_map.count("config-key")) ||
                (!variable_map.count("config-file") && variable_map.count("config-key"))) {
            printUsage(visible_options);
            std::cout << "Error: config file must be supplied with a corresponding config-key. Exiting.\n";
            return -1;
        } else if (variable_map.count("config-file")) {
            config_used = true;
        }

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type!" << std::endl;
    }

    // Create the specified TYPE of detector
    Detector* detector;

    switch (type_hash[type]) {
        case 'a':
        {
            detector = new DifferenceDetector(source, sink);
            break;
        }
        case 'b':
        {
            detector = new HSVDetector(source, sink);
            break;
        }
        default:
        {
            printUsage(options);
            std::cout << "Error: invalid TYPE specified. Exiting.\n";
            return -1;
        }
    }

    if (config_used)
        detector->configure(config_file, config_key);

    // Two threads - one for user interaction, the other
    // for executing the processor
    boost::thread_group thread_group;
    thread_group.create_thread(boost::bind(&run, detector));
    sleep(1);

    std::cout << "Detector has begun listening to source \"" + source + "\".\n";
    std::cout << "Detector has begun steaming to sink \"" + sink + "\".\n\n";
    std::cout << "COMMANDS:\n";
    std::cout << "  t: Enable tuning mode.\n";
    std::cout << "  T: Disable tuning mode.\n";
    std::cout << "  x: Exit.\n";

    while (!done) {

        char user_input;
        std::cin >> user_input;

        switch (user_input) {

            case 't':
            {
                detector->set_tune_mode(true);
                break;
            }
            case 'T':
            {
                detector->set_tune_mode(false);
                break;
            }
            case 'x':
            {
                done = true;
                detector->stop();
                break;
            }
            default:
                std::cout << "Invalid command. Try again.\n";
                break;
        }

    }

    // TODO: Exit gracefully and ensure all shared resources are cleaned up!
    thread_group.join_all();

    // Free heap memory allocated to detector 
    delete detector;
    
    std::cout << "Detector is exiting.\n";

    // Exit
    return 0;
}


