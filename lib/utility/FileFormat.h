//******************************************************************************
//* File:   FileFormat.h
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

#ifndef OAT_FILEFORMAT_H
#define	OAT_FILEFORMAT_H

#include <string>

namespace oat {

/**
 * Create and verify save path for various files used within the Oat project.
 *
 * @param save_path_result Formated and verified save path
 * @param save_directory Requested save directory
 * @param base_file_name Requested base file name. Extension should be included.
 * @param prepend_timestamp Should a timestamp be prepended to the file name?
 * @param allow_overwrite In case of conflict, should file be overwritten?
 * @return Errorcode: -1 : Unspecified failure
 *                     0 : Success
 *                     1 : Requested save directory does not exist
 *                     2 : User does not have write permissions to the file.
 *                     3 : File name was empty
 */
int createSavePath(std::string &save_path_result,
                   const std::string &save_directory,
                   const std::string &base_file_name,
                   const bool prepend_timestamp,
                   const bool allow_overwrite);

/**
 * Generate a current timestamp formated as Y-M-D-H-M-S.
 *
 * @return Timestamp.
 */
std::string createTimeStamp();

/**
 * Ensure that a file path is unique. If a file with the same path is
 * encountered file path is appended with a numeral.
 *
 * @param file_path Full path of the file to check and potentially modify
 * @return True if the file_path was modified due to a conflict.
 */
bool ensureUniquePath(std::string& file_path);

/**
 * Check if we have write access to a file.
 *
 * @param file_path Full path of file to check
 * @return True if we have write permission, false otherwise.
 */
bool checkWritePermission(const std::string &file_path);

}      /* namespace oat */
#endif /* OAT_FILEFORMAT_H */

