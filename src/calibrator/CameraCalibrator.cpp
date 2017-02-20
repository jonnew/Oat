//******************************************************************************
//* File:   CameraCalibrator.cpp
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

#include "CameraCalibrator.h"
#include "PathChanger.h"
#include "Saver.h"
#include "UsagePrinter.h"

#include <map>
#include <utility>

#include <boost/io/ios_state.hpp>
#include <cpptoml.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/IOUtility.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

const std::vector<std::string> CameraCalibrator::mode_strings_{
    "Normal", "Detect", "Undistort"};

CameraCalibrator::CameraCalibrator(const std::string &source_name) :
  Calibrator(source_name)
{
    // Initialize corner detection update timers
    tick_ = Clock::now();
    tock_ = Clock::now();

#ifdef HAVE_OPENGL
    try {
        cv::namedWindow(name(), cv::WINDOW_OPENGL & cv::WINDOW_KEEPRATIO);
    } catch (cv::Exception& ex) {
        oat::whoWarn(name(), "OpenCV not compiled with OpenGL support."
                "Falling back to OpenCV's display driver.\n");
        cv::namedWindow(name(), cv::WINDOW_NORMAL & cv::WINDOW_KEEPRATIO);
    }
#else
    cv::namedWindow(name(), cv::WINDOW_NORMAL & cv::WINDOW_KEEPRATIO);
#endif

    std::cout << "Starting interactive session.\n";
}

po::options_description CameraCalibrator::options() const
{
    // Common program options
    po::options_description local_opts;
    local_opts.add_options()
        ("calibration-key,k", po::value<std::string>(),
        "The key name for the calibration entry that will be inserted "
        "into the calibration file. e.g. 'camera-1-homography'\n")
        ("calibration-path,f", po::value<std::string>(),
        "The calibration file location. If not is specified,"
        "defaults to './calibration.toml'. If a folder is specified, "
        "defaults to '<folder>/calibration.toml\n. If a full path "
        "including file in specified, then it will be that path "
        "without modification.")
        ("chessboard-size,s", po::value<std::string>(),
        "Int array, [x,y], specifying the number of inside corners in "
        "the horizontal and vertical demensions of the chessboard used for calibration.\n")
        ("square-width,w", po::value<double>(),
        "The length/width of a single chessboard square in meters.\n")
        ;

    return local_opts;
}

void CameraCalibrator::applyConfiguration(
    const po::variables_map &vm, const config::OptionTable &config_table)
{
    oat::config::getValue<std::string>(
        vm, config_table, "calibration-key", calibration_key_);

    oat::config::getValue<std::string>(
        vm, config_table, "calibration-path", calibration_save_path_);

    generateSavePath(calibration_save_path_, "calibration");

    // Square width (must come before chessboard size because it is used there)
    oat::config::getNumericValue<double>(vm, config_table, "square-width",
                                         square_size_meters_, 0);

    // Chessboard size
    std::vector<double> s;
    if (oat::config::getArray<double, 2>(vm, config_table, "chessboard-size", s, true)) {

        if (s[0] <= 2 || s[1] <= 2)
            throw(std::runtime_error(
                "Chessboard width and height should be greater than 2."));

        chessboard_size_.height = s[0];
        chessboard_size_.width  = s[1];

        // Generate true corner locations based upon the chessboard size and
        // square size
        for (int i = 0; i < chessboard_size_.height; i++) {
            for (int j = 0; j < chessboard_size_.width; j++) {
                corners_meters_.push_back(
                        cv::Point3f(static_cast<double>(j) * square_size_meters_,
                                    static_cast<double>(i) * square_size_meters_,
                                    0.0f));
            }
        }
    }
}

void CameraCalibrator::calibrate(cv::Mat &frame)
{
    tick_ = Clock::now();

    // TODO: Is it possible to get frame metadata before any loops? Might
    // be hard if client (this) strarts first since the frame source is not
    // yet known and therefore things like frame size are not known.
    frame_size_ = frame.size();

    if (mode_ == Mode::DETECT)
        detectChessboard(frame);

    if (mode_ == Mode::UNDISTORT && calibration_valid_)
        undistortFrame(frame);

    // Add mode and status info
    decorateFrame(frame);

    cv::imshow(name(), frame);
    char command = cv::waitKey(1);

    switch (command) {

        case 'c': // Clear all data points
        {
            if (requireMode(std::forward<Mode>(Mode::NORMAL)))
                clearDataPoints();
            break;
        }
        case 'd': // Enter/exit chessboard corner capture mode
        {
            if (requireMode(std::forward<Mode>(Mode::NORMAL), std::forward<Mode>(Mode::DETECT)))
                toggleMode(Mode::DETECT);
            break;
        }
        case 'f': // Change the calibration save path
        {
            if (requireMode(std::forward<Mode>(Mode::NORMAL))) {
                PathChanger changer;
                accept(&changer);
            }
            break;
        }
        case 'g': // Generate calibration parameters
        {
            if (requireMode(std::forward<Mode>(Mode::NORMAL)))
                generateCalibrationParameters();
            break;
        }
        case 'h': // Display help dialog
        {
            UsagePrinter usage;
            accept(&usage, std::cout);
            break;
        }
        case 'p': // Print calibration results
        {
            printCalibrationResults(std::cout);
            break;
        }
        case 'u': // Undistort mode
        {
            if (requireMode(std::forward<Mode>(Mode::NORMAL),
                            std::forward<Mode>(Mode::UNDISTORT)))
                toggleMode(Mode::UNDISTORT);
            break;
        }
        case 's': // Save homography info
        {
            Saver saver(calibration_key_, calibration_save_path_);
            accept(&saver);
            break;
        }
    }
}

void CameraCalibrator::accept(CalibratorVisitor *visitor)
{
    visitor->visit(this);
}

void CameraCalibrator::accept(OutputVisitor *visitor, std::ostream &out)
{
    visitor->visit(this, out);
}

void CameraCalibrator::clearDataPoints()
{
    corners_.clear();
}

void CameraCalibrator::detectChessboard(cv::Mat &frame)
{
    // Extract the chessboard from the current image
    std::vector<cv::Point2f> point_buffer;
    bool detected =
        cv::findChessboardCorners(frame, chessboard_size_, point_buffer,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

    // Draw corners on the frame
    cv::drawChessboardCorners(frame, chessboard_size_, cv::Mat(point_buffer), detected);

    // Calculate elapsed time since last detection
    if (detected) {

        Milliseconds elapsed_time =
            std::chrono::duration_cast<Milliseconds>(tick_ - tock_);

        if (elapsed_time > min_detection_delay_) {

            std::cout << "Chessboard detected.\n";

            // Reset timer
            tock_ = Clock::now();

            // Subpixel corner location estimation termination criteria
            // Max iterations = 30;
            // Desired accuracy of pixel resolution = 0.1
            cv::TermCriteria term(
                    cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1);

            // Generate grey-scale image
            cv::Mat frame_grey;
            cv::cvtColor(frame, frame_grey, cv::COLOR_BGR2GRAY);

            // Find exact corner locations
            cv::cornerSubPix(frame_grey, point_buffer, cv::Size(11, 11),
                    cv::Size(-1, -1), term);

            // Push the new corners into storage
            corners_.push_back(point_buffer);

            // Note visually that we have added new corners to our data set
            cv::bitwise_not(frame, frame);
        }
    }
}

void CameraCalibrator::undistortFrame(cv::Mat &frame)
{
    // NOTE: For cv::undistort, src must not be the same Mat as dest.
    cv::Mat temp = frame.clone();
    cv::undistort(temp, frame, camera_matrix_, distortion_coefficients_);
}

void CameraCalibrator::generateCalibrationParameters()
{
    if (corners_.size() == 0) {
        std::cerr << oat::Error("At least one chessboard detection "
                                "is needed to generate calibration "
                                "parameters.\n");
        return;
    }

    // Intermediates required by the calibration routines
    // but not used elsewhere currently
    std::vector<std::vector<cv::Point3f>> object_points;
    object_points.resize(corners_.size(), corners_meters_);

    // Reset the calibration settings
    int calibration_flags = 0;

    // Reinitialized the camera matrix and distortion coefficients
    // with sizes required for pinhole model
    camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
    distortion_coefficients_ = cv::Mat::zeros(8, 1, CV_64F);

    // TODO: user options for the following
    // Fix the aspect ratio of the lens (ratio of lens focal lengths for each
    // dimension of its internal reference frame, fc(2)/fc(1) where fc =
    // [KK[1,1]; KK[2,2])
    // calibration_flags |= CALIB_FIX_ASPECT_RATIO;

    // Set tangential distortion coefficients (last three elements of KC) to
    // zero. This is reasonable for modern cameras that have very good
    // centering of lens over the sensory array.
    // calibration_flags |= CALIB_ZERO_TANGENT_DIST

    // Make principle point cc = [KK[3,1]; KK[3,2]] equal to the center of the
    // frame : cc = [(nx-1)/2;(ny-1)/2)]
    // calibration_flags |= CALIB_FIX_PRINCIPAL_POINT

    rms_error_ = cv::calibrateCamera(
            object_points,
            corners_,
            frame_size_,
            camera_matrix_,
            distortion_coefficients_,
            cv::noArray(),
            cv::noArray(),
            calibration_flags | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5);

    calibration_valid_ =
            cv::checkRange(camera_matrix_) && cv::checkRange(distortion_coefficients_);

    if (calibration_valid_)
        printCalibrationResults(std::cout);
}

void CameraCalibrator::decorateFrame(cv::Mat &frame)
{
    auto m_idx = static_cast<typename std::underlying_type<Mode>::type>(mode_);
    std::string mode_msg =  "Mode: " + mode_strings_[m_idx];
    cv::Size txt_size = cv::getTextSize(mode_msg, 1, 1, 1, 0);
    cv::Point mode_origin(frame.cols - txt_size.width - 10, frame.rows - 10);
    cv::putText(frame, mode_msg, mode_origin, 1, 1, cv::Scalar(0, 0, 255));
}

void CameraCalibrator::printCalibrationResults(std::ostream &out)
{
    // Save stream state. When ifs is destructed, the stream will
    // return to default format.
    boost::io::ios_flags_saver ifs(out);

    // Data set information
    out << "CAMERA CALIBRATION RESULTS\n"
        << "Number of corner location lists in data set: " <<  corners_.size() << "\n\n";

    // Calibration results information
    out << "Calibration Valid: " << std::boolalpha << calibration_valid_ << "\n\n"
        << "Camera Matrix:\n"
        << camera_matrix_ << "\n\n"
        << "Distortion Coefficients:\n"
        << distortion_coefficients_ << "\n\n"
        << "RMS Reconstruction Error: " << rms_error_ << "\n\n";
}

void CameraCalibrator::toggleMode(Mode mode)
{
    if (mode_ != mode)
        mode_ = mode;
    else
        mode_ = Mode::NORMAL;
}

} /* namespace oat */
