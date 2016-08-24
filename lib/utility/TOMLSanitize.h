//******************************************************************************
//* File:   OatTOMLSanitize.h
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

#ifndef OAT_TOMLSANATIZE_H
#define OAT_TOMLSANATIZE_H

#include <iostream>
#include <limits>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <boost/program_options.hpp>
#include <boost/type_index.hpp>

#include <cpptoml.h>

namespace oat{
namespace config {

namespace po = boost::program_options;

// Aliases for TOML table and array types
using Value = std::shared_ptr<cpptoml::base>;
using OptionTable = std::shared_ptr<cpptoml::table>;
using Array = std::shared_ptr<cpptoml::array>;

using OptionMap = boost::program_options::variables_map;
using boost::typeindex::type_id_with_cvr;

inline std::string
noTableError(const std::string& table_name,
             const std::string& config_file) {

    return  "No configuration table named '" + table_name +
            "' was provided in the configuration file '" + config_file + "'";
}

inline std::string
valueError(const std::string& entry_name,
           const std::string& table_name,
           const std::string& config_file,
           const std::string& message) {

    return "'" + entry_name + "' in '" + table_name + "' in '" + config_file + "' " + message;
}

/** @brief Extract a configuration table from a config file. The OptionMap must
 * contain an entry specifying a file/key pair. File must be a path to a valid
 * TOML file. Entries under key parameter within the TOML file can be used
 * within inidividual components to specify runtime parameters but should be
 * checked against valid program options prior to use.
 *
 * @param map Program option map extacted form CLI input @param key Key within
 * option map specifying file/key pair of config file.
 */
inline const OptionTable
getConfigTable(const OptionMap map, const char *key="config") {

    std::vector<std::string> fk_pair;

    if (!map[key].empty()) {

        fk_pair = map[key].as<std::vector<std::string> >();

        if (fk_pair.size() != 2)
           throw std::runtime_error("Configuration must be supplied as file key pair.");
    }

    if (fk_pair.empty())
        return cpptoml::make_table();

    // Will throw if file contains bad syntax
    auto config = cpptoml::parse_file(fk_pair[0]);

    if (!config->contains(fk_pair[1])) {
        throw (std::runtime_error(
            oat::config::noTableError(fk_pair[1], fk_pair[0])
        ));
    }

    // Get this components configuration table
    return config->get_table(fk_pair[1]);
}

inline void
checkKeys(const std::vector<std::string> &options,
          const OptionTable user_config) {

    auto it = user_config->begin();

    // Look through user-provided config options and make sure each matches an
    // actual configuration key ID
    while (it != user_config->end()) {

        auto key = it->first;

        if (std::find(std::begin(options), std::end(options), key) == options.end()) {
            throw (std::runtime_error("Unknown configuration key '" + key + "'."));
        }

        it++;
    }
}

/**
 * @brief Retrieve nested table with and unspecified number of elements from an
 * exsiting table.
 *
 * @param table Parent table
 * @param key Key of child table.
 * @param nested_table Child table.
 *
 * @return True if child table was successfully assigned.
 */
inline bool
getTable(const OptionTable table,
         const std::string &key,
         OptionTable &nested_table) {

    // If the key is in the table,
    if (table->contains(key)) {

        // Make sure the key points to a value (and not a table, array, or table-array)
        if (table->get(key)->is_table()) {

            // Get value
            nested_table = table->get_table(key);
            return true;

        } else {
            throw (std::runtime_error("'" + key + "' must be a TOML table."));
        }

    } else {
        return false;
    }
}

/**
 * @brief Retrieve program option either from command line map or from config
 * file. Preference is given to command line. Additionally, perform type
 * sanitation.
 *
 * @param vm Program option variable map obtained from command line input.
 * @param table Potentially empty table generated from a TOML config file.
 * @param key OptionTable key of value to retrieve.
 * @param value Resulting value.
 * @param required Specifies whether a value must be specified either via
 * command line or config file.
 *
 * @return True if value was successfully defined.
 */
template <typename T>
bool getValue(const po::variables_map &vm,
              const OptionTable table,
              const std::string& key,
              T &value,
              bool required = false) {

    if (vm.count(key)) {

        value = vm[key].as<T>();
        return true;

    } else if (table->contains(key)) {

        // Make sure the key points to a value (and not a table, array, or table-array)
        if (table->get(key)->is_value()) {

            // Get value
            value = *table->get_as<T>(key);
            return true;

        } else {
            throw (std::runtime_error("'" + key + "' must be a TOML value of type "
                    + type_id_with_cvr<T>().pretty_name() + "."));
        }

    } else if (required) {
         throw (std::runtime_error("Required configuration value '" + key + "' was not specified."));
    } else {
        return false;
    }
}

/**
 * @brief Retrieve program option either from command line map or from config
 * file. Preference is given to command line. Additionally, perform type
 * sanitation.
 *
 * @param vm Program option variable map obtained from command line input.
 * @param table Potentially empty table generated from a TOML config file.
 * @param key OptionTable key of value to retrieve.
 * @param value Resulting value.
 * @param required Specifies whether a value must be specified either via
 * command line or config file.
 *
 * @return True if value was successfully defined.
 */
template <typename T>
bool
getNumericValue(const po::variables_map &vm,
                const OptionTable table,
                const std::string &key,
                T &value,
                const T lower = std::numeric_limits<T>::min(),
                const T upper = std::numeric_limits<T>::max(),
                bool required = false) {

    static_assert (std::is_integral<T>::value ||
                   std::is_floating_point<T>::value, "Numeric type required.");

    if (vm.count(key)) {

        value = vm[key].as<T>();

        // Get value, check range
        if (value < lower || value > upper) {
            throw (std::runtime_error("Configuration key '" + key +
                    "' specifies a value that is out of bounds."));
        }

        return true;

    } else if (table->contains(key)) {

        // Make sure the key points to a value (and not a table, array, or table-array)
        if (table->get(key)->is_value()) {

            // CPP TOML only works with int64_t
            if (std::is_integral<T>::value) {
                auto val = *(table->get_as<int64_t>(key));
                value = static_cast<T>(val);
            } else if (std::is_floating_point<T>::value) {
                auto val = *(table->get_as<double>(key));
                value = static_cast<T>(val);
            } else {
                value = *(table->get_as<double>(key));
            }

            // Get value, check range
            if (value < lower || value > upper) {
                throw (std::runtime_error("Configuration key '" + key +
                        "' specifies a value that is out of bounds."));
            }

            return true;

        } else {
            throw (std::runtime_error("'" + key + "' must be a TOML value of type "
                    + type_id_with_cvr<T>().pretty_name() + "."));
        }

    } else if (required) {
         throw (std::runtime_error("Required configuration value '" + key + "' was not specified."));
    } else {
        return false;
    }
}

// TOML array from table, any size
inline bool
getArray(const OptionTable table,
         const std::string& key,
         Array &array_out,
         bool required = false) {

    // If the key is in the table,
    if (table->contains(key)) {

        // Make sure the key points to a value (and not a table, array, or table-array)
        if (table->get(key)->is_array()) {

            array_out = table->get_array(key);
            return true;

        } else {
            throw (std::runtime_error("'" + key + "' must be a TOML array."));
        }
    } else if (required) {
         throw (std::runtime_error("Required configuration value '" + key + "' was not specified."));
    } else {
        return false;
    }
}

// TOML array from table, required size
inline bool
getArray(const OptionTable table,
         const std::string& key,
         Array &array_out,
         size_t size,
         bool required = false) {

    // If the key is in the table,
    if (table->contains(key)) {

        // Make sure the key points to a value (and not a table, array, or table-array)
        if (table->get(key)->is_array()) {

            array_out = table->get_array(key);

            if (array_out->get().size() != size) {
                throw (std::runtime_error("'" + key + "' must be a TOML vector "
                        "containing " + std::to_string(size) + " elements."));
            }

            return true;

        } else {
            throw (std::runtime_error("'" + key + "' must be a TOML vector "
                    "containing " + std::to_string(size) + " elements."));
        }
    } else if (required) {
        throw (std::runtime_error("Required configuration value '" + key + "' was not specified."));
    } else {
        return false;
    }
}

}      /* namespace config */
}      /* namespace oat */
#endif /* OAT_TOMLSANATIZE_H */
