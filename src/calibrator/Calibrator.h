//******************************************************************************
//* File:   Calibrator.h
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
//****************************************************************************

#ifndef OAT_CALIBRATOR_H
#define OAT_CALIBRATOR_H

#include <string>
#include <iosfwd>

#include <boost/program_options.hpp>
#include <opencv2/core/mat.hpp>

#include "../../lib/datatypes/Frame.h"
#include "../../lib/shmemdf/Source.h"

namespace po = boost::program_options;

namespace oat {

// Forward decl.
class CalibratorVisitor;
class OutputVisitor;
class SharedFrameHeader;

/**
 * Abstract calibrator.
 * All concrete calibrator types implement this ABC.
 */
class Calibrator {

public:

    /**
     * Abstract calibrator.
     * All concrete calibrator types implement this ABC.
     * @param source_address Frame SOURCE address
     */
    Calibrator(const std::string &source_address);

    virtual ~Calibrator() {};

    /**
     * Calibrator SOURCE must be able to connect to a NODE from
     * which to receive frames.
     */
    virtual void connectToNode(void);

    /**
     * Run the calibration routine on the frame SOURCE.
     * @return True if SOURCE signals EOF
     */
    virtual bool process(void);

    /**
     * @brief Append type-specific program options.
     * @param opts Program option description to be specialized.
     */
    virtual void appendOptions(po::options_description &opts);

    /**
     * @brief Configure filter parameters.
     * @param vm Previously parsed program option value map.
     */
    virtual void configure(const po::variables_map &vm);

    /**
     * Create the calibration file path using a specified path.
     * @param save_path the path to save configuration data. If this path is a
     * folder, the calibration file will default to calibraiton.toml.
     * @param default_name Default file name in the case that a full path is
     * not provided.
     * @return True if the specified calibration file already exists.
     */
    virtual bool generateSavePath(const std::string &save_path,
                                  const std::string &default_name);

    // Accept functions for visitors
    virtual void accept(CalibratorVisitor* visitor) = 0;
    virtual void accept(OutputVisitor* visitor, std::ostream& out) = 0;

    // Accessors
    //const std::string & name() const { return name_; }
    std::string calibration_save_path() const {
        return calibration_save_path_;
    }

    void set_calibration_key(const std::string& value) {
        calibration_key_ = value;
    }

    std::string name() const { return name_; }

protected:

    // List of allowed configuration options, including those
    // specified only via config file
    std::vector<std::string> config_keys_;

    /** Perform calibration routine.
     * @param frame frame to use for generating calibration parameters
     */
    virtual void calibrate(cv::Mat& frame) = 0;

    std::string calibration_key_ {"calibration"};  //!< Key name of calibration table entry
    std::string calibration_save_path_ {"."};      //!< Calibration parameter save path

private:

    std::string name_;                  //!< Calibrator name
    oat::Frame internal_frame_;         //!< Current frame provided by SOURCE
    std::string source_address_;        //!< Frame source address
    oat::Source<Frame> frame_source_;   //!< The calibrator frame SOURCE

    //std::string calibration_key_;       //!< Key name of calibration table entry
    //std::string calibration_save_path_; //!< Calibration parameter save path
};

}      /* namespace oat */
#endif /* OAT_CALIBRATOR_H */
