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

#include "HSVDetector.h"

#include <signal.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

volatile sig_atomic_t done = 0;
bool running = true;

void term(int) {
    done = 1;
}

void run(HSVDetector* detector, std::string source, std::string sink) {
    
    std::cout << "HSV detector \"" + detector->get_detector_name() + "\" has begun listening to source \"" + source + "\".\n";
    std::cout << "HSV detector \"" + detector->get_detector_name() + "\" has begun steaming to sink \"" + sink + "\".\n";

    while (!done) { // !done
        if (running) {
            detector->applyFilterAndServe();
        }
    }
    
    std::cout << "HSV detector \"" + detector->get_detector_name() + "\" is exiting.\n";
}

void printUsage(po::options_description options) {
    std::cout << "Usage: hsvdetector [OPTIONS]\n";
    std::cout << "   or: hsvdetector SOURCE SINK [IMAGE SINK] [CONFIGURATION]\n";
    std::cout << "Perform HSV object detection on images from SMServer<SharedCVMatHeader> SOURCE.\n";
    std::cout << "Publish detected object positions to a SMSserver<Position2D> SINK.\n";
    std::cout << options << "\n";
}

int main(int argc, char *argv[]) {

    signal(SIGINT, term);
    
    std::string source;
    std::string sink;
    std::string frame_sink;
    std::string config_file;
    std::string config_key;
    bool config_used = false;
    bool frame_sink_used = false;

    try {

        po::options_description options("OPTIONS");
        options.add_options()
                ("help", "Produce help message.")
                ("version,v", "Print version information.")
                ;

        po::options_description imgsink("IMAGE SINK");
        imgsink.add_options()
                ("framesink", po::value<std::string>(&frame_sink),"Optional IMAGE SINK SMServer<SharedCVMatHeader> to which processed images will be published.")
                ;
        
        po::options_description config("CONFIGURATION");
        config.add_options()
                ("config-file,c", po::value<std::string>(&config_file), "Configuration file.")
                ("config-key,k", po::value<std::string>(&config_key), "Configuration key.")
                ;

        po::options_description hidden("HIDDEN OPTIONS");
        hidden.add_options()
                ("source", po::value<std::string>(&source),
                "The name of the SOURCE that supplies images on which hsv-filter object detection will be performed."
                "The server must be of type SMServer<SharedCVMatHeader>\n")
                ("sink", po::value<std::string>(&sink),
                "The name of the SINK to which position background subtracted images will be served.")
                ;

        po::positional_options_description positional_options;
        positional_options.add("source", 1);
        positional_options.add("sink", 2);

        po::options_description visible_options("VISIBLE OPTIONS");
        visible_options.add(options).add(imgsink).add(config);

        po::options_description all_options("ALL OPTIONS");
        all_options.add(options).add(imgsink).add(config).add(hidden);

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
        
        if (variable_map.count("framesink")) {
            frame_sink_used = true;
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

    HSVDetector detector(source, sink);
    
    if (config_used)
        detector.configure(config_file, config_key);
    
    if (frame_sink_used)
        detector.addFrameSink(frame_sink);
    
    // Two threads - one for user interaction, the other
    // for executing the processor
    boost::thread_group thread_group;
    thread_group.create_thread(boost::bind(&run, &detector, source, sink));
    sleep(1);

    while (!done) {
        int user_input;
        std::cout << "Select an action:\n";
        std::cout << " [1]: Pause/unpause\n";
        std::cout << " [2]: Exit\n";
        std::cout << ">> ";

        std::cin >> user_input;

        switch (user_input) {

            case 1:
            {
                running = !running;
                if (running)
                    std::cout << " Resumed...\n";
                else
                    std::cout << " Paused.\n";
                break;
            }

            case 2:
            {
                done = true;
                detector.stop();
                break;
            }
            default:
                std::cout << "Invalid selection. Try again.\n";
                break;
        }
    }
    
    // TODO: Exit gracefully and ensure all shared resources are cleaned up!
    thread_group.join_all();

    // Exit
    return 0;
}


