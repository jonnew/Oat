//******************************************************************************
//* File:   FileReader.cpp
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

#include "FileReader.h"

#include <thread>

#include <cpptoml.h>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

FileReader::FileReader(const std::string &sink_address)
: FrameServer(sink_address)
{
    // Initialize time
    tick_ = clock_.now();
}

po::options_description FileReader::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("video-file,f", po::value<std::string>(),
         "Path to video file to serve frames from.")
        ("rate,r", po::value<double>(),
         "Rate, in FPS, to read the video. Defaults to as fast as possible.")
        ("bounds,b", po::value<std::string>(),
         "Two element array of ints, [first,last] which specify the "
         "first and last frame to read in the file. last = -1 incates the file "
         "should be read to the end. Defaults to the complete file. ")
        ("skip,s", po::value<int>(),
         "Number of frames to skip between reads. Defaults to 0.")
        ("roi", po::value<std::string>(),
         "Four element array of unsigned ints, [x0,y0,width,height],"
         "defining a rectangular region of interest. Origin "
         "is upper left corner. ROI must fit within acquired "
         "frame size. Defaults to full video size.")
        ;

    return local_opts;
}

void FileReader::applyConfiguration(const po::variables_map &vm,
                                    const config::OptionTable &config_table)
{
    // Video file
    std::string file_name;
    oat::config::getValue(vm, config_table, "video-file", file_name, true);
    file_reader_.open(file_name);

    // Read rate
    double r;
    if (oat::config::getNumericValue(vm, config_table, "rate", r, 0.0))
        read_period_ = Token::Seconds(1.0 / r);

    // Bounds
    oat::config::getArray<int, 2>(vm, config_table, "bounds", bounds_);

    // Skip number
    oat::config::getNumericValue(vm, config_table, "skip", skip_, 0);

    // ROI
    std::vector<size_t> roi;
    if (oat::config::getArray<size_t, 4>(vm, config_table, "roi", roi)) {

        use_roi_ = true;
        region_of_interest_.x      = roi[0];
        region_of_interest_.y      = roi[1];
        region_of_interest_.width  = roi[2];
        region_of_interest_.height = roi[3];
    }
}

bool FileReader::connectToNode()
{
    // Get sample frame
    cv::Mat example_frame;
    file_reader_ >> example_frame;

    // (Re)set the start position
    file_reader_.set(CV_CAP_PROP_POS_FRAMES, bounds_[0]);

    if (use_roi_)
        example_frame = example_frame(region_of_interest_);

    // Get the native video frame rate
    auto fps = file_reader_.get(CV_CAP_PROP_FPS);

    frame_sink_.reserve(example_frame.total() * example_frame.elemSize());
    frame_sink_.bind(Token::Seconds(static_cast<double>(skip_ + 1) / fps),
                     example_frame.rows,
                     example_frame.cols,
                     color_);

    // Link shared_frame_ to shmem storage
    shared_frame_ = frame_sink_.retrieve();

    return true;
}

int FileReader::process()
{
    cv::Mat mat;
    if (!file_reader_.read(mat)
        || (bounds_[1] != -1
            && file_reader_.get(CV_CAP_PROP_POS_FRAMES) > bounds_[1]))
        return 1;

    // Skip frames if needed
    if (skip_ > 0) {
        auto pos = file_reader_.get(CV_CAP_PROP_POS_FRAMES);
        file_reader_.set(CV_CAP_PROP_POS_FRAMES, pos + skip_);
    }

    if (use_roi_)
        mat = mat(region_of_interest_);

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    frame_sink_.wait();

    shared_frame_->incrementCount(skip_);
    shared_frame_->copyFrom(mat);

    // Tell sources there is new data
    frame_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    std::this_thread::sleep_for(read_period_ - (clock_.now() - tick_));
    tick_ = clock_.now();

    return 0;
}

} /* namespace oat */
