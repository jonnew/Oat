//******************************************************************************
//* File:   FileFormat.cpp
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

#include <ctime>
#include <string>
#include <boost/filesystem.hpp>

#include "FileFormat.h"

namespace oat {

namespace bfs = boost::filesystem;

int createSavePath(std::string &save_path_result,
                   const std::string &save_directory,
                   const std::string &base_file_name,
                   const std::string &prepend_str, 
                   const bool use_prepend_str,
                   const bool allow_overwrite) {

    // First check that the save_directory is valid
    bfs::path path(save_directory.c_str());
    if (!bfs::exists(path) || !bfs::is_directory(path))
        return 1;

    // Make sure there is a file_name
    if (base_file_name.empty())
        return 3;

    // Now that the directory has been confirmed, created file name
    // Create a single position file
    if (use_prepend_str)
        save_path_result = save_directory + "/" + prepend_str + base_file_name;
    else
        save_path_result = save_directory + "/" + base_file_name;

    if (!allow_overwrite)
       ensureUniquePath(save_path_result);

    if (!checkWritePermission(save_path_result))
        return 2;

    return 0;
}

/**
 * Generate a current timestamp formated as Y-M-D-H-M-S.
 * @return
 */
std::string createTimeStamp() {

    std::time_t raw_time;
    struct tm * time_info;
    char buffer[100];
    std::time(&raw_time);
    time_info = std::localtime(&raw_time);
    std::strftime(buffer, 80, "%F-%H-%M-%S", time_info);

    return std::string(buffer);
}


bool ensureUniquePath(std::string& file) {

    int i = 0;
    std::string original_file = file;

    while (bfs::exists(file.c_str())) {

        ++i;
        bfs::path path(original_file.c_str());
        bfs::path parent_path = path.parent_path();
        bfs::path stem = path.stem();
        bfs::path extension = path.extension();

        std::string append = "_" + std::to_string(i);
        stem += append.c_str();

        // Recreate file name
        file = std::string(parent_path.generic_string()) + "/" +
               std::string(stem.generic_string()) +
               std::string(extension.generic_string());
    }

    return (i != 0) ? true : false; 
}

bool checkWritePermission(const std::string &file_path) {

    FILE *fp = fopen(file_path.c_str(), "w");
    bool can_write = fp != nullptr;
    if (can_write)
        fclose(fp);

    return  can_write;
}

}      /* namespace oat */
