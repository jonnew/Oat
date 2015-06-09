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

// *************** NOTE *********************
// This program is basically a copy of the OpenCV camera calibration tutorial
// found here: https://github.com/Itseez/opencv/blob/master/samples/cpp/calibration.cpp
// Except that it works with simple-tracker's Camera interface.


#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

#include "Camera.h"
#include "PGGigECam.h"
#include "WebCam.h"
#include "FileReader.h"

using namespace cv;
using namespace std;

const char * usage =
        " \nexample command line for calibration from a live feed.\n"
        "   calibration  -w 4 -h 5 -s 0.025 -o camera.yml -op -oe\n"
        " \n"
        " example command line for calibration from a list of stored images:\n"
        "   imagelist_creator image_list.xml *.png\n"
        "   calibration -w 4 -h 5 -s 0.025 -o camera.yml -op -oe image_list.xml\n"
        " where image_list.xml is the standard OpenCV XML/YAML\n"
        " use imagelist_creator to create the xml or yaml list\n"
        " file consisting of the list of strings, e.g.:\n"
        " \n"
        "<?xml version=\"1.0\"?>\n"
        "<opencv_storage>\n"
        "<images>\n"
        "view000.png\n"
        "view001.png\n"
        "<!-- view002.png -->\n"
        "view003.png\n"
        "view010.png\n"
        "one_extra_view.jpg\n"
        "</images>\n"
        "</opencv_storage>\n";


const char* liveCaptureHelp =
        "When the live video from camera is used as input, the following hot-keys may be used:\n"
        "  <ESC>, 'q' - quit the program\n"
        "  'g' - start capturing images\n"
        "  'u' - switch undistortion on/off\n"
        "  'w' - world view calibration\n";

static void help() {
    printf("This is a camera calibration sample.\n"
            "Usage: calibration\n"
            ""
            "     -w <board_width>         # the number of inner corners per one of board dimension\n"
            "     -h <board_height>        # the number of inner corners per another board dimension\n"
            "     [-pt <pattern>]          # the type of pattern: chessboard or circles' grid\n"
            "     [-n <number_of_frames>]  # the number of frames to use for calibration\n"
            "                              # (if not specified, it will be set to the number\n"
            "                              #  of board views actually available)\n"
            "     [-d <delay>]             # a minimum delay in ms between subsequent attempts to capture a next view\n"
            "                              # (used only for video capturing)\n"
            "     [-s <squareSize>]        # square size in some user-defined units (1 by default)\n"
            "     [-o <out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"
            "     [-op]                    # write detected feature points\n"
            "     [-oe]                    # write extrinsic parameters\n"
            "     [-zt]                    # assume zero tangential distortion\n"
            "     [-a <aspectRatio>]       # fix aspect ratio (fx/fy)\n"
            "     [-p]                     # fix the principal point at the center\n"
            "     [-v]                     # flip the captured images around the horizontal axis\n"
            "     [-t]                     # Camera type: \'list\', \'wcam\', \'gige\', or \'file\'\n"
            "                              # [input_data] string for the video file name\n"
            "     [-c]                     # Camera configuration file\n"
            "     [-k]                     # Camera configuration key\n"
            "     [-su]                    # show undistorted images after calibration\n"
            "     [-D <distance>]          # Standard distance used for world unit conversion (e.g. 1.0 meters)"
            "     [input_data]             # input data, one of the following:\n"
            "                              #  - (-t list) text file with a list of the images of the board\n"
            "                              #    the text file can be generated with imagelist_creator\n"
            "                              #  - (-t file) name of video file with a video of the board\n"
            "                              # if input_data not specified, a live view from the camera is used\n"
            "\n");
    printf("\n%s", usage);
    printf("\n%s", liveCaptureHelp);
}

enum {
    DETECTION = 0, CAPTURING = 1, CALIBRATED = 2, WORLDCOORD = 3
};

enum Pattern {
    CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID
};

enum CameraType {
    LIST, WCAM, GIGE, VIDFILE
};

static double computeReprojectionErrors(
        const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs,
        vector<float>& perViewErrors) {
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for (i = 0; i < (int) objectPoints.size(); i++) {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int) objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err * err / n);
        totalErr += err*err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD) {
    corners.resize(0);

    switch (patternType) {
        case CHESSBOARD:
        case CIRCLES_GRID:
            for (int i = 0; i < boardSize.height; i++)
                for (int j = 0; j < boardSize.width; j++)
                    corners.push_back(Point3f(float(j * squareSize),
                        float(i * squareSize), 0));
            break;

        case ASYMMETRIC_CIRCLES_GRID:
            for (int i = 0; i < boardSize.height; i++)
                for (int j = 0; j < boardSize.width; j++)
                    corners.push_back(Point3f(float((2 * j + i % 2) * squareSize),
                        float(i * squareSize), 0));
            break;

        default:
            CV_Error(Error::StsBadArg, "Unknown pattern type\n");
    }
}

static bool runCalibration(vector<vector<Point2f> > imagePoints,
        Size imageSize, Size boardSize, Pattern patternType,
        float squareSize, float aspectRatio,
        int flags, Mat& cameraMatrix, Mat& distCoeffs,
        vector<Mat>& rvecs, vector<Mat>& tvecs,
        vector<float>& reprojErrs,
        double& totalAvgErr) {
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if (flags & CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0, 0) = aspectRatio;

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);

    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
            distCoeffs, rvecs, tvecs, flags | CALIB_FIX_K4 | CALIB_FIX_K5);
    ///*|CALIB_FIX_K3*/|CALIB_FIX_K4|CALIB_FIX_K5);
    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
            rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}

static void saveCameraParams(const string& filename,
        Size imageSize, Size boardSize,
        float squareSize, float aspectRatio, int flags,
        const Mat& cameraMatrix, const Mat& distCoeffs,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const vector<float>& reprojErrs,
        const vector<vector<Point2f> >& imagePoints,
        double totalAvgErr,
        bool homography_valid, cv::Mat homography2D) {
    FileStorage fs(filename, FileStorage::WRITE);

    time_t tt;
    time(&tt);
    struct tm *t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof (buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;

    if (!rvecs.empty() || !reprojErrs.empty())
        fs << "nframes" << (int) std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if (flags & CALIB_FIX_ASPECT_RATIO)
        fs << "aspectRatio" << aspectRatio;

    if (flags != 0) {
        sprintf(buf, "flags: %s%s%s%s",
                flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
                flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
                flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
                flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
        //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if (!reprojErrs.empty())
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if (!rvecs.empty() && !tvecs.empty()) {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int) rvecs.size(), 6, rvecs[0].type());
        for (int i = 0; i < (int) rvecs.size(); i++) {
            Mat r = bigmat(Range(i, i + 1), Range(0, 3));
            Mat t = bigmat(Range(i, i + 1), Range(3, 6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }

    if (!imagePoints.empty()) {
        Mat imagePtMat((int) imagePoints.size(), (int) imagePoints[0].size(), CV_32FC2);
        for (int i = 0; i < (int) imagePoints.size(); i++) {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }

    if (homography_valid) {
        fs << "homography" << homography2D;
    }
}

static bool readStringList(const string& filename, vector<string>& l) {
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != FileNode::SEQ)
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it)
        l.push_back((string) * it);
    return true;
}

static bool runAndSave(const string& outputFilename,
        const vector<vector<Point2f> >& imagePoints,
        Size imageSize,
        Size boardSize,
        Pattern patternType,
        float squareSize,
        float aspectRatio,
        int flags,
        Mat& cameraMatrix,
        Mat& distCoeffs,
        bool writeExtrinsics,
        bool writePoints,
        bool homography_valid,
        cv::Mat homography2D) {
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
            aspectRatio, flags, cameraMatrix, distCoeffs,
            rvecs, tvecs, reprojErrs, totalAvgErr);
    printf("%s. avg reprojection error = %.2f\n",
            ok ? "Calibration succeeded" : "Calibration failed",
            totalAvgErr);

    if (ok)
        saveCameraParams(outputFilename,
            imageSize,
            boardSize,
            squareSize,
            aspectRatio,
            flags,
            cameraMatrix,
            distCoeffs,
            writeExtrinsics ? rvecs : vector<Mat>(),
            writeExtrinsics ? tvecs : vector<Mat>(),
            writeExtrinsics ? reprojErrs : vector<float>(),
            writePoints ? imagePoints : vector<vector<Point2f> >(),
            totalAvgErr,
            homography_valid,
            homography2D);
    return ok;
}

static void mouseEvent(int event, int x, int y, int flags, void* ptr) {

    if (event == EVENT_LBUTTONDOWN) {
        Point* p = (Point*) ptr;
        p->x = x;
        p->y = y;

        cout << "Position (" << x << ", " << y << ")\n";
    }
}

int main(int argc, char** argv) {
    Size boardSize, imageSize;
    float squareSize = 1.f, aspectRatio = 1.f;
    Mat cameraMatrix, distCoeffs;
    const char* outputFilename = "out_camera_data.yml";
    const char* inputFilename = 0;
    std::string dummy_sink = "dummy_sink";

    int i, nframes = 10;
    bool writeExtrinsics = false, writePoints = false;
    bool undistortImage = false;
    int flags = 0;

    bool use_simple_tracker_camera = false;
    Camera* camera; // TODO: add new flag to determine if this is used and what type it is
    std::string config_file;
    std::string config_key;
    bool config_used = false;
    float standard_distance_in_world_units;
    bool show_world_coords = false;

    bool homography_valid = false;
    cv::Mat homography2D;


    bool flipVertical = false;
    bool showUndistorted = false;
    //bool videofile = false;
    int delay = 1000;
    clock_t prevTimestamp = 0;
    int mode = DETECTION;
    int cameraId = 0;
    vector<vector<Point2f> > imagePoints;
    vector<string> imageList;
    Pattern pattern = CHESSBOARD;
    CameraType camera_type = WCAM;

    if (argc < 2) {
        help();
        return 0;
    }

    for (i = 1; i < argc; i++) {
        const char* s = argv[i];
        if (strcmp(s, "-w") == 0) {
            if (sscanf(argv[++i], "%u", &boardSize.width) != 1 || boardSize.width <= 0)
                return fprintf(stderr, "Invalid board width\n"), -1;
        } else if (strcmp(s, "-h") == 0) {
            if (sscanf(argv[++i], "%u", &boardSize.height) != 1 || boardSize.height <= 0)
                return fprintf(stderr, "Invalid board height\n"), -1;
        } else if (strcmp(s, "-t") == 0) {
            i++;
            if (!strcmp(argv[i], "list"))
                camera_type = LIST;
            else if (!strcmp(argv[i], "wcam"))
                camera_type = WCAM;
            else if (!strcmp(argv[i], "gige"))
                camera_type = GIGE;
            else if (!strcmp(argv[i], "file"))
                camera_type = VIDFILE;
            else
                return fprintf(stderr, "Invalid camera type: must be list, wcam, gige, or file\n"), -1;
        } else if (strcmp(s, "-pt") == 0) {
            i++;
            if (!strcmp(argv[i], "circles"))
                pattern = CIRCLES_GRID;
            else if (!strcmp(argv[i], "acircles"))
                pattern = ASYMMETRIC_CIRCLES_GRID;
            else if (!strcmp(argv[i], "chessboard"))
                pattern = CHESSBOARD;
            else
                return fprintf(stderr, "Invalid pattern type: must be chessboard or circles\n"), -1;
        } else if (strcmp(s, "-c") == 0) {
            config_file = argv[++i];
            config_used = true;
        } else if (strcmp(s, "-k") == 0) {
            config_key = argv[++i];
            config_used = true;
        } else if (strcmp(s, "-s") == 0) {
            if (sscanf(argv[++i], "%f", &squareSize) != 1 || squareSize <= 0)
                return fprintf(stderr, "Invalid board square width\n"), -1;
        } else if (strcmp(s, "-n") == 0) {
            if (sscanf(argv[++i], "%u", &nframes) != 1 || nframes <= 3)
                return printf("Invalid number of images\n"), -1;
        } else if (strcmp(s, "-a") == 0) {
            if (sscanf(argv[++i], "%f", &aspectRatio) != 1 || aspectRatio <= 0)
                return printf("Invalid aspect ratio\n"), -1;
            flags |= CALIB_FIX_ASPECT_RATIO;
        } else if (strcmp(s, "-d") == 0) {
            if (sscanf(argv[++i], "%u", &delay) != 1 || delay <= 0)
                return printf("Invalid delay\n"), -1;
        } else if (strcmp(s, "-op") == 0) {
            writePoints = true;
        } else if (strcmp(s, "-oe") == 0) {
            writeExtrinsics = true;
        } else if (strcmp(s, "-zt") == 0) {
            flags |= CALIB_ZERO_TANGENT_DIST;
        } else if (strcmp(s, "-p") == 0) {
            flags |= CALIB_FIX_PRINCIPAL_POINT;
        } else if (strcmp(s, "-v") == 0) {
            flipVertical = true;
        } else if (strcmp(s, "-o") == 0) {
            outputFilename = argv[++i];
        } else if (strcmp(s, "-D") == 0) {
            standard_distance_in_world_units = std::stof(argv[++i]);
        } else if (strcmp(s, "-su") == 0) {
            showUndistorted = true;
        } else if (s[0] != '-') {
            if (isdigit(s[0]))
                sscanf(s, "%d", &cameraId);
            else
                inputFilename = s;
        } else
            return fprintf(stderr, "Unknown option %s", s), -1;
    }

    switch (camera_type) {
        case LIST:
            readStringList(inputFilename, imageList);
            nframes = (int) imageList.size();
            mode = CAPTURING;
            break;

        case WCAM:
            use_simple_tracker_camera = true;
            camera = new WebCam(dummy_sink);
            break;

        case GIGE:
            use_simple_tracker_camera = true;
            camera = new PGGigECam(dummy_sink);
            break;

        case VIDFILE:
            use_simple_tracker_camera = true;
            camera = new FileReader(inputFilename, dummy_sink);
            break;

        default:
            return fprintf(stderr, "Unknown camera type \n"), -2;
    }

    if (config_used && config_file.empty() || config_used && config_key.empty()) {
        std::cerr << "Error: Camera config file must be supplied with a corresponding config-key. Exiting.\n";
        return -1;
    }

    if (use_simple_tracker_camera && config_used)
        camera->configure(config_file, config_key);
    else if (use_simple_tracker_camera)
        camera->configure();

    if (use_simple_tracker_camera)
        printf("%s", liveCaptureHelp);

    namedWindow("Image View", 1);

    //set the callback function for any mouse event
    cv::Point mouse_pt;
    setMouseCallback("Image View", mouseEvent, &mouse_pt);

    for (i = 0;; i++) {
        Mat view, viewGray;
        bool blink = false;

        if (use_simple_tracker_camera) {
            Mat view0;
            camera->grabMat();
            view0 = camera->getCurrentFrame();
            view0.copyTo(view);
        } else if (i < (int) imageList.size())
            view = imread(imageList[i], 1);

        if (view.empty()) {
            if (imagePoints.size() > 0)
                runAndSave(outputFilename, imagePoints, imageSize,
                    boardSize, pattern, squareSize, aspectRatio,
                    flags, cameraMatrix, distCoeffs,
                    writeExtrinsics, writePoints,
                    homography_valid,
                    homography2D);
            break;
        }

        imageSize = view.size();

        if (flipVertical)
            flip(view, view, 0);

        vector<Point2f> pointbuf;
        cvtColor(view, viewGray, COLOR_BGR2GRAY);

        bool found;
        switch (pattern) {
            case CHESSBOARD:
                found = findChessboardCorners(view, boardSize, pointbuf,
                        CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
                break;
            case CIRCLES_GRID:
                found = findCirclesGrid(view, boardSize, pointbuf);
                break;
            case ASYMMETRIC_CIRCLES_GRID:
                found = findCirclesGrid(view, boardSize, pointbuf, CALIB_CB_ASYMMETRIC_GRID);
                break;
            default:
                return fprintf(stderr, "Unknown pattern type\n"), -1;
        }

        // improve the found corners' coordinate accuracy
        if (pattern == CHESSBOARD && found) cornerSubPix(viewGray, pointbuf, Size(11, 11),
                Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

        if (mode == CAPTURING && found &&
                (!use_simple_tracker_camera || clock() - prevTimestamp > delay * 1e-3 * CLOCKS_PER_SEC)) {
            imagePoints.push_back(pointbuf);
            prevTimestamp = clock();
            blink = use_simple_tracker_camera;
        }

        if (found)
            drawChessboardCorners(view, boardSize, Mat(pointbuf), found);

        string msg = mode == CAPTURING ? "100/100" :
                mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

        if (mode == CAPTURING) {
            if (undistortImage)
                msg = format("%d/%d Undist", (int) imagePoints.size(), nframes);
            else
                msg = format("%d/%d", (int) imagePoints.size(), nframes);
        }

        if (blink)
            bitwise_not(view, view);

        if (mode == CALIBRATED && undistortImage) {
            Mat temp = view.clone();
            undistort(temp, view, cameraMatrix, distCoeffs);
        }

        int key = 0xff & waitKey(use_simple_tracker_camera ? 50 : 500);

        if ((key & 255) == 27)
            break;

        if (key == 'u' && mode == CALIBRATED)
            undistortImage = !undistortImage;

        // Decorate image after undistort
        if (use_simple_tracker_camera && key == 'g') {
            mode = CAPTURING;
            imagePoints.clear();
        }

        putText(view, msg, textOrigin, 1, 1,
                mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));


        imshow("Image View", view);

        if (key == 'w') {
            mode = WORLDCOORD;
        }

        if (mode == CAPTURING && imagePoints.size() >= (unsigned) nframes) {
            if (runAndSave(outputFilename, imagePoints, imageSize,
                    boardSize, pattern, squareSize, aspectRatio,
                    flags, cameraMatrix, distCoeffs,
                    writeExtrinsics, writePoints,
                    homography_valid,
                    homography2D))
                mode = CALIBRATED;
            else
                mode = DETECTION;
            if (!use_simple_tracker_camera)
                break;
        }

        // This is a complete hack for my specific purposes
        if (mode == WORLDCOORD) {

            if (!undistortImage) {
                std::cout << "Warning: generating world coordinates in reference to a distorted image!!\n";
                std::cout << "You might want to undistort the image before generating the world reference frame.\n";
                mode = CALIBRATED;

            }

            std::cout << "Homography calculation mode started.\n";
            std::cout << "Hot keys:"
                    << "  'x' - exit homography generation routine and attempt to calculate translation matrix.\n"
                    << "  'c' - calculate homography matrix using current points.\n"
                    << "  'a' - add currently selected pixel position to homography coordinate list.\n\n";

            std::cout << "Click a points on the frozen image. You will see the pixel value at the terminal.\n";
            std::cout << "When you are satisfied with the point's placement, press 'a' to continue...\n";


            std::vector<cv::Point2f> src_points;
            std::vector<cv::Point2f> dst_points;

            bool done = false;

            while (!done) {

                // Plot the current mouse click position
                cv::Mat view_with_dot = view.clone();
                cv::circle(view_with_dot, mouse_pt, 2, cv::Scalar(0, 0, 255), -1);

                int baseLine = 0;
                std::string coord = "(" + std::to_string(mouse_pt.x) + ", " + std::to_string(mouse_pt.y) + ")";
                //Size coord_text_size = cv::getTextSize(coord, 1, 1, 1, &baseLine);
                Point coord_text_origin(mouse_pt.x + 10.0, mouse_pt.y + 10.0);
                putText(view_with_dot, coord, coord_text_origin, 1, 1, Scalar(0, 0, 255));

                if (homography_valid) {

                    std::vector<cv::Point2f> Q_world;
                    std::vector<cv::Point2f> q_camera;

                    q_camera.push_back(mouse_pt);

                    cv::perspectiveTransform(q_camera, Q_world, homography2D);

                    std::string coord = "(" + std::to_string(Q_world[0].x)
                            + ", " + std::to_string(Q_world[0].y) + ")";

                    Point coord_text_origin(mouse_pt.x + 10.0, mouse_pt.y - 10.0);
                    putText(view_with_dot, coord, coord_text_origin, 1, 1, Scalar(0, 0, 255));
                }

                cv::imshow("Image View", view_with_dot);

                int key = 0xff & waitKey(50);

                switch (key) {

                    case 'x':
                    {
                        done = true;
                        break;
                    }

                    case 'c':
                    {
                        if (src_points.size() > 1 && dst_points.size() > 1) {
                            //homography_transform = cv::findHomography(src_points, dst_points);
                            homography2D = cv::estimateRigidTransform(src_points, dst_points, true);
                            cv::Mat row = (Mat_<double>(1, 3) << 0, 0, 1);
                            homography2D.push_back(row);
                            std::cout << "T = " << endl << " " << homography2D << "\n\n";
                            homography_valid = true;
                        } else {
                            std::cout << "Add more points to before calculating.\n";
                        }
                        break;
                    }

                    case 'a':
                    {

                        try {
                            float x, y;
                            cv::Point2f src_pt;
                            cv::Point2f dst_pt;

                            std::string input_coords;

                            std::cout << "Enter X world coordinate:\n";
                            std::cin >> input_coords;
                            x = std::stof(input_coords);

                            std::cout << "Enter Y world coordinate:\n";
                            std::cin >> input_coords;
                            y = std::stof(input_coords);

                            src_pt.x = (float) mouse_pt.x;
                            src_pt.y = (float) mouse_pt.y;
                            dst_pt.x = x;
                            dst_pt.y = y;

                            src_points.push_back(src_pt);
                            dst_points.push_back(dst_pt);

                            std::cout << "Point added to map. Select another action:\n";

                        } catch (std::invalid_argument ex) {
                            std::cout << "Invalid input.\n";
                        }

                        std::cout << "Hot keys:"
                                << "  'x' - exit homography generation routine and attempt to calculate translation matrix.\n"
                                << "  'c' - calculate homography matrix using current points.\n"
                                << "  'a' - add currently selected pixel position to homography coordinate list.\n\n";
                        break;
                    }
                    default:
                        //std::cout << "Invalid command. Try again.\n";
                        break;
                }
            }

            if (homography_valid) {

                runAndSave(outputFilename, imagePoints, imageSize,
                        boardSize, pattern, squareSize, aspectRatio,
                        flags, cameraMatrix, distCoeffs,
                        writeExtrinsics, writePoints,
                        homography_valid,
                        homography2D);
            }

            mode = CALIBRATED;

        }
    }

    if (!use_simple_tracker_camera && showUndistorted) {
        Mat view, rview, map1, map2;
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                imageSize, CV_16SC2, map1, map2);

        for (i = 0; i < (int) imageList.size(); i++) {
            view = imread(imageList[i], 1);
            if (view.empty())
                continue;
            //undistort( view, rview, cameraMatrix, distCoeffs, cameraMatrix );
            remap(view, rview, map1, map2, INTER_LINEAR);
            imshow("Image View", rview);
            int c = 0xff & waitKey();
            if ((c & 255) == 27 || c == 'q' || c == 'Q')
                break;
        }
    }

    return 0;
}