//******************************************************************************
//* File:   IOFormat.h
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

#ifndef IOFORMAT_H
#define	IOFORMAT_H

#include <string>

namespace oat {

    inline std::string bold(std::string message) {

        return "\033[1m" + message + "\033[0m";
    }

    inline std::string sourceText(std::string source_name) {

        return "\033[32m" + source_name + "\033[0m";
    }

    inline std::string sinkText(std::string sink_name) {

        return "\033[31m" + sink_name + "\033[0m";
    }

    inline std::string whoMessage(std::string source, std::string message) {

        return "\033[1m" + source + ": \033[0m" + message;
    }

    inline std::string whoWarn(std::string source, std::string message) {

        return "\033[1m" + source + ": \033[0m\033[33m" + message + "\033[0m";
    }
    
    inline std::string Warn(std::string message) {

        return "\033[0m\033[33m" + message + "\033[0m";
    }

    inline std::string whoError(std::string source, std::string message) {

        return "\033[1m" + source + ": \033[0m\033[31m" + message + "\033[0m";
    }
    
    inline std::string configNoTableError(std::string table_name,
                                          std::string config_file) {

        return  "No configuration table named '" + table_name +
                "' was provided in the configuration file '" + config_file + "'";
    }

    inline std::string configValueError(std::string entry_name,
                                        std::string table_name,
                                        std::string config_file,
                                        std::string message) {

        return "'" + entry_name + "' in '" + table_name + "' in '" + config_file + "' " + message;
    }
    
    inline std::string Error(std::string message) {

        return "\033[31m" + message + "\033[0m";
    }
    
    inline std::string dbgMessage(std::string message) {

        return "\033[35mdebug: " + message + "\033[0m";
    }
    
    inline std::string dbgColor(std::string message) {

        return "\033[35m" + message + "\033[0m";
    }
}

#endif	/* IOFORMAT_H */
