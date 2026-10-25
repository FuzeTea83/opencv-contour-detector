#ifndef OPENCV_INTERACTIVE_H
#define OPENCV_INTERACTIVE_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "stats.h"
#include "sobel.h"
#include "laplacian.h"
#include "canny.h"

namespace Interactive
{
enum DetectorType
{
    DETECTOR_SOBEL     = 0,
    DETECTOR_LAPLACIAN = 1,
    DETECTOR_CANNY     = 2,
    DETECTOR_COUNT     = 3
};

struct ViewerState
{
    const std::vector<cv::Mat> &images;
    const std::vector<cv::Mat> &references;
    const std::vector<std::string> &filenames;

    std::string winName = "Interactive Edge Detector";

    int trackbarImage = 0;
    int trackbarDetector = 0;     // 0: Sobel, 1: Laplacian, 2: Canny
    int trackbarBlur = 3;         // 0, 1, 3, 5, 7, 9, 11, 13, 15
    int trackbarKsizeIdx = 1;     // 0: 1, 1: 3, 2: 5, 3: 7
    int trackbarThreshold1 = 50;  // Threshold (Sobel/Lap) or Threshold1 (Canny)
    int trackbarThreshold2 = 100; // Threshold2 (Canny)
    int trackbarL2 = 0;           // 0: L1, 1: L2 (Canny)
    int trackbarScale = 10;       // Scale x10 (10 -> 1.0)
    int trackbarDelta = 0;        // Delta (0..100)

    bool updatingTrackbars = false;
    bool needsRedraw = true;

    ViewerState(const std::vector<cv::Mat> &imgs,
                const std::vector<cv::Mat> &refs,
                const std::vector<std::string> &names)
        : images(imgs), references(refs), filenames(names) {}

    int getActualBlurSize() const
    {
        if (trackbarBlur <= 0) return 0;
        int b = trackbarBlur;
        if (b % 2 == 0) b += 1;
        return b;
    }

    int getActualKsize() const
    {
        static constexpr int ksizes[] = {1, 3, 5, 7};
        const int idx = std::clamp(trackbarKsizeIdx, 0, 3);
        return ksizes[idx];
    }

    int getActualCannyAperture() const
    {
        static constexpr int apertures[] = {3, 5, 7};
        const int idx = std::clamp(trackbarKsizeIdx, 0, 2);
        return apertures[idx];
    }

    void setFromSobelParams(const Sobel::Params &p)
    {
        updatingTrackbars = true;
        trackbarBlur = p.blurSize;
        trackbarKsizeIdx = (p.ksize == 1) ? 0 : (p.ksize == 3) ? 1 : (p.ksize == 5) ? 2 : 3;
        trackbarThreshold1 = std::clamp(p.threshold, 0, 255);
        trackbarScale = std::clamp(static_cast<int>(p.scale * 10.0), 1, 50);
        trackbarDelta = std::clamp(static_cast<int>(p.delta), 0, 100);
        syncTrackbars();
        updatingTrackbars = false;
    }

    void setFromLaplacianParams(const Laplacian::Params &p)
    {
        updatingTrackbars = true;
        trackbarBlur = p.blurSize;
        trackbarKsizeIdx = (p.ksize == 1) ? 0 : (p.ksize == 3) ? 1 : (p.ksize == 5) ? 2 : 3;
        trackbarThreshold1 = std::clamp(p.threshold, 0, 255);
        trackbarScale = std::clamp(static_cast<int>(p.scale * 10.0), 1, 50);
        trackbarDelta = std::clamp(static_cast<int>(p.delta), 0, 100);
        syncTrackbars();
        updatingTrackbars = false;
    }

    void setFromCannyParams(const Canny::Params &p)
    {
        updatingTrackbars = true;
        trackbarBlur = p.blurSize;
        trackbarKsizeIdx = (p.apertureSize == 3) ? 0 : (p.apertureSize == 5) ? 1 : 2;
        trackbarThreshold1 = std::clamp(p.threshold1, 0, 255);
        trackbarThreshold2 = std::clamp(p.threshold2, 0, 255);
        trackbarL2 = p.L2gradient ? 1 : 0;
        syncTrackbars();
        updatingTrackbars = false;
    }

    void syncTrackbars() const
    {
        cv::setTrackbarPos("Image", winName, trackbarImage);
        cv::setTrackbarPos("Detector", winName, trackbarDetector);
        cv::setTrackbarPos("Blur", winName, trackbarBlur);
        cv::setTrackbarPos("Kernel", winName, trackbarKsizeIdx);
        cv::setTrackbarPos("Threshold 1", winName, trackbarThreshold1);
        cv::setTrackbarPos("Threshold 2", winName, trackbarThreshold2);
        cv::setTrackbarPos("L2 Norm", winName, trackbarL2);
        cv::setTrackbarPos("Scale x10", winName, trackbarScale);
        cv::setTrackbarPos("Delta", winName, trackbarDelta);
    }

    void optimizeCurrent()
    {
        if (images.empty() || trackbarImage >= static_cast<int>(images.size())) return;

        const cv::Mat &img = images[trackbarImage];
        const cv::Mat &ref = references[trackbarImage];

        if (trackbarDetector == DETECTOR_SOBEL)
        {
            const auto [p, s] = Sobel::findBestParams(img, ref);
            setFromSobelParams(p);
        }
        else if (trackbarDetector == DETECTOR_LAPLACIAN)
        {
            const auto [p, s] = Laplacian::findBestParams(img, ref);
            setFromLaplacianParams(p);
        }
        else if (trackbarDetector == DETECTOR_CANNY)
        {
            const auto [p, s] = Canny::findBestParams(img, ref);
            setFromCannyParams(p);
        }
        needsRedraw = true;
    }
};

static void onTrackbarChange(int, void *userdata)
{
    auto *state = static_cast<ViewerState *>(userdata);
    if (!state->updatingTrackbars) { state->needsRedraw = true; }
}

/**
 * @brief Builds a colored error map to visualize detection quality.
 * Green = True Positive (correctly detected edge)
 * Red = False Positive (extra detected edge)
 * Blue = False Negative (missed edge compared to ground truth)
 * White = Background
 */
static cv::Mat buildErrorMap(const cv::Mat &det, const cv::Mat &ref)
{
    const int rows = det.rows;
    const int cols = det.cols;
    cv::Mat errorMap(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
    std::vector<uint8_t> refMatched(rows * cols, 0);

    for (int y = 0; y < rows; ++y)
    {
        const auto *ptrDet = det.ptr<uchar>(y);
        const int yMin = (y > 0) ? (y - 1) : 0;
        const int yMax = (y + 1 < rows) ? (y + 1) : (rows - 1);

        for (int x = 0; x < cols; ++x)
        {
            if (ptrDet[x] != 0) continue;

            const int xMin = (x > 0) ? (x - 1) : 0;
            const int xMax = (x + 1 < cols) ? (x + 1) : (cols - 1);

            bool matched = false;
            for (int ny = yMin; ny <= yMax && !matched; ++ny)
            {
                const auto *ptrRef = ref.ptr<uchar>(ny);
                for (int nx = xMin; nx <= xMax; ++nx)
                {
                    if (ptrRef[nx] == 0 && !refMatched[ny * cols + nx])
                    {
                        refMatched[ny * cols + nx] = 1;
                        matched = true;
                        break;
                    }
                }
            }

            if (matched)
            {
                errorMap.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 180, 0); // Green = True Positive
            }
            else
            {
                errorMap.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 230); // Red = False Positive
            }
        }
    }

    // Mark false negatives (ground truth edges not found)
    for (int y = 0; y < rows; ++y)
    {
        const auto *ptrRef = ref.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x)
        {
            if (ptrRef[x] == 0 && !refMatched[y * cols + x])
            {
                errorMap.at<cv::Vec3b>(y, x) = cv::Vec3b(220, 80, 0); // Blue = False Negative
            }
        }
    }

    return errorMap;
}

/**
 * @brief Draws an information and metrics banner above the view.
 */
static cv::Mat createInfoBanner(int width,
                                const std::string &imageName,
                                int imgIdx, int totalImgs,
                                const std::string &detName,
                                const std::string &paramsStr,
                                const Stats &stats)
{
    constexpr int bannerHeight = 90;
    cv::Mat banner(bannerHeight, width, CV_8UC3, cv::Scalar(35, 38, 48));

    std::ostringstream line1, line2, line3;
    line1 << "Image [" << (imgIdx + 1) << "/" << totalImgs << "] : " << imageName
            << "   |   Detector : " << detName;

    line2 << "Parameters : " << paramsStr;

    line3 << std::fixed << std::setprecision(4)
            << "(P) : " << stats.P
            << "   |   False Positives (FPR) : " << stats.FPR
            << "   |   False Negatives (FNR) : " << stats.FNR
            << "   |   F1-Score : " << stats.F1;

    const double fontScale = std::clamp(width / 1600.0 * 0.9, 0.55, 0.85);

    cv::putText(banner, line1.str(), cv::Point(20, 26), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(245, 245, 245),
                1, cv::LINE_AA);
    cv::putText(banner, line2.str(), cv::Point(20, 52), cv::FONT_HERSHEY_SIMPLEX, fontScale * 0.92,
                cv::Scalar(130, 220, 255), 1, cv::LINE_AA);
    cv::putText(banner, line3.str(), cv::Point(20, 78), cv::FONT_HERSHEY_SIMPLEX, fontScale * 0.95,
                cv::Scalar(110, 255, 150), 1, cv::LINE_AA);

    return banner;
}

/**
 * @brief Full rendering of the interactive interface and display.
 */
static void renderViewer(ViewerState &state)
{
    if (state.images.empty()) return;

    state.trackbarImage = std::clamp(state.trackbarImage, 0, static_cast<int>(state.images.size()) - 1);
    state.trackbarDetector = std::clamp(state.trackbarDetector, 0, static_cast<int>(DETECTOR_COUNT) - 1);

    const int imgIdx = state.trackbarImage;
    const cv::Mat &srcImg = state.images[imgIdx];
    const cv::Mat &refImg = state.references[imgIdx];
    const std::string &fileName = state.filenames[imgIdx];

    cv::Mat detectedContour;
    std::string detectorName;
    std::string paramsSummary;

    const int blur = state.getActualBlurSize();

    if (state.trackbarDetector == DETECTOR_SOBEL)
    {
        detectorName = "Sobel";
        Sobel::Params p;
        p.ksize = state.getActualKsize();
        p.blurSize = blur;
        p.threshold = state.trackbarThreshold1;
        p.scale = std::max(0.1, state.trackbarScale / 10.0);
        p.delta = state.trackbarDelta;

        detectedContour = Sobel::contour(srcImg, p);

        std::ostringstream ss;
        ss << "ksize=" << p.ksize << ", blur=" << p.blurSize
                << ", threshold=" << p.threshold << ", scale=" << p.scale << ", delta=" << p.delta;
        paramsSummary = ss.str();
    }
    else if (state.trackbarDetector == DETECTOR_LAPLACIAN)
    {
        detectorName = "Laplacian (LoG)";
        Laplacian::Params p;
        p.ksize = state.getActualKsize();
        p.blurSize = blur;
        p.threshold = state.trackbarThreshold1;
        p.scale = std::max(0.1, state.trackbarScale / 10.0);
        p.delta = state.trackbarDelta;

        detectedContour = Laplacian::contour(srcImg, p);

        std::ostringstream ss;
        ss << "ksize=" << p.ksize << ", blur=" << p.blurSize
                << ", threshold=" << p.threshold << ", scale=" << p.scale << ", delta=" << p.delta;
        paramsSummary = ss.str();
    }
    else
    {
        detectorName = "Canny";
        Canny::Params p;
        p.apertureSize = state.getActualCannyAperture();
        p.blurSize = blur;
        p.threshold1 = state.trackbarThreshold1;
        p.threshold2 = state.trackbarThreshold2;
        p.L2gradient = (state.trackbarL2 != 0);

        detectedContour = Canny::contour(srcImg, p);

        std::ostringstream ss;
        ss << "T1=" << p.threshold1 << ", T2=" << p.threshold2
                << ", aperture=" << p.apertureSize << ", blur=" << p.blurSize
                << ", L2=" << (p.L2gradient ? "true" : "false");
        paramsSummary = ss.str();
    }

    const Stats stats = calculateStats(detectedContour, refImg);

    // Prepare 4 BGR views
    cv::Mat viewSrc, viewDet, viewRef, viewErr;

    if (srcImg.channels() == 1) cv::cvtColor(srcImg, viewSrc, cv::COLOR_GRAY2BGR);
    else viewSrc = srcImg.clone();

    cv::cvtColor(detectedContour, viewDet, cv::COLOR_GRAY2BGR);
    cv::cvtColor(refImg, viewRef, cv::COLOR_GRAY2BGR);
    viewErr = buildErrorMap(detectedContour, refImg);

    // Titles on each view
    auto addTitle = [](cv::Mat &img, const std::string &title, const cv::Scalar &color)
    {
        const int barH = std::max(28, static_cast<int>(img.rows * 0.08));
        cv::rectangle(img, cv::Point(0, 0), cv::Point(img.cols, barH), cv::Scalar(25, 25, 25), cv::FILLED);
        cv::putText(img, title, cv::Point(12, barH - 8), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv::LINE_AA);
    };

    addTitle(viewSrc, "1. Original Image", cv::Scalar(255, 255, 255));
    addTitle(viewDet, "2. Detection (" + detectorName + ")", cv::Scalar(0, 255, 255));
    addTitle(viewRef, "3. Ground Truth", cv::Scalar(200, 200, 200));
    addTitle(viewErr, "4. Analysis (G=TP, R=FP, B=FN)", cv::Scalar(100, 255, 100));

    // Use high resolution to avoid upscaling blur
    const int targetH = std::max(srcImg.rows, 380);
    const int targetW = static_cast<int>(static_cast<double>(targetH) / srcImg.rows * srcImg.cols);

    cv::Mat rSrc, rDet, rRef, rErr;
    if (viewSrc.rows == targetH && viewSrc.cols == targetW)
    {
        rSrc = viewSrc;
        rDet = viewDet;
        rRef = viewRef;
        rErr = viewErr;
    }
    else
    {
        cv::resize(viewSrc, rSrc, cv::Size(targetW, targetH), 0, 0, cv::INTER_LINEAR);
        cv::resize(viewDet, rDet, cv::Size(targetW, targetH), 0, 0, cv::INTER_NEAREST);
        cv::resize(viewRef, rRef, cv::Size(targetW, targetH), 0, 0, cv::INTER_NEAREST);
        cv::resize(viewErr, rErr, cv::Size(targetW, targetH), 0, 0, cv::INTER_NEAREST);
    }

    // 2x2 grid concatenation
    cv::Mat topRow, botRow, mosaic;
    cv::hconcat(rSrc, rDet, topRow);
    cv::hconcat(rRef, rErr, botRow);
    cv::vconcat(topRow, botRow, mosaic);

    // Upper info banner
    const cv::Mat banner = createInfoBanner(mosaic.cols, fileName, imgIdx,
                                            static_cast<int>(state.images.size()),
                                            detectorName, paramsSummary, stats);

    // Lower help bar
    constexpr int helpHeight = 32;
    cv::Mat helpBar(helpHeight, mosaic.cols, CV_8UC3, cv::Scalar(28, 28, 32));
    const std::string helpText =
            "[N/P or Arrows]: Next/Prev image   |   [S/L/C]: Sobel/Laplacian/Canny   |   [O]: Auto-optimize   |   [Q/ESC]: Quit";
    cv::putText(helpBar, helpText, cv::Point(20, 21), cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(190, 190, 190), 1,
                cv::LINE_AA);

    cv::Mat fullDisplay;
    cv::vconcat(banner, mosaic, fullDisplay);
    cv::vconcat(fullDisplay, helpBar, fullDisplay);

    cv::imshow(state.winName, fullDisplay);
}

/**
 * @brief Main loop of the interactive viewer.
 */
static void run(const std::vector<cv::Mat> &images,
                const std::vector<cv::Mat> &references,
                const std::vector<std::string> &filenames,
                const int initialDetector = DETECTOR_CANNY)
{
    if (images.empty()) return;

    ViewerState state(images, references, filenames);
    state.trackbarDetector = initialDetector;

    cv::namedWindow(state.winName, cv::WINDOW_NORMAL);
    cv::resizeWindow(state.winName, 1024, 768);

    // Create trackbars with dedicated callbacks (prevents OpenCV warnings)
    cv::createTrackbar("Image", state.winName, nullptr,
                       static_cast<int>(images.size()) - 1,
                       [](int pos, void *ud)
                       {
                           auto *s = static_cast<ViewerState *>(ud);
                           if (!s->updatingTrackbars)
                           {
                               s->trackbarImage = pos;
                               s->optimizeCurrent();
                           }
                       }, &state);

    cv::createTrackbar("Detector", state.winName, nullptr,
                       DETECTOR_COUNT - 1,
                       [](const int pos, void *ud)
                       {
                           auto *s = static_cast<ViewerState *>(ud);
                           if (!s->updatingTrackbars)
                           {
                               s->trackbarDetector = pos;
                               s->optimizeCurrent();
                           }
                       }, &state);

    cv::createTrackbar("Blur", state.winName, nullptr,
                       15,
                       [](const int pos, void *ud)
                       {
                           auto *s = static_cast<ViewerState *>(ud);
                           s->trackbarBlur = pos;
                           if (!s->updatingTrackbars) s->needsRedraw = true;
                       }, &state);

    cv::createTrackbar("Kernel", state.winName, nullptr,
                       3,
                       [](const int pos, void *ud)
                       {
                           auto *s = static_cast<ViewerState *>(ud);
                           s->trackbarKsizeIdx = pos;
                           if (!s->updatingTrackbars) s->needsRedraw = true;
                       }, &state);

    cv::createTrackbar("Threshold 1", state.winName, nullptr,
                       255,
                       [](const int pos, void *ud)
                       {
                           auto *s = static_cast<ViewerState *>(ud);
                           s->trackbarThreshold1 = pos;
                           if (!s->updatingTrackbars) s->needsRedraw = true;
                       }, &state);

    cv::createTrackbar("Threshold 2", state.winName, nullptr,
                       255,
                       [](const int pos, void *ud)
                       {
                           auto *s = static_cast<ViewerState *>(ud);
                           s->trackbarThreshold2 = pos;
                           if (!s->updatingTrackbars) s->needsRedraw = true;
                       }, &state);

    cv::createTrackbar("L2 Norm", state.winName, nullptr,
                       1,
                       [](const int pos, void *ud)
                       {
                           auto *s = static_cast<ViewerState *>(ud);
                           s->trackbarL2 = pos;
                           if (!s->updatingTrackbars) s->needsRedraw = true;
                       }, &state);

    cv::createTrackbar("Scale x10", state.winName, nullptr,
                       50,
                       [](const int pos, void *ud)
                       {
                           auto *s = static_cast<ViewerState *>(ud);
                           s->trackbarScale = pos;
                           if (!s->updatingTrackbars) s->needsRedraw = true;
                       }, &state);

    cv::createTrackbar("Delta", state.winName, nullptr,
                       100,
                       [](const int pos, void *ud)
                       {
                           auto *s = static_cast<ViewerState *>(ud);
                           s->trackbarDelta = pos;
                           if (!s->updatingTrackbars) s->needsRedraw = true;
                       }, &state);

    // Initial optimization on the first image
    state.optimizeCurrent();

    std::cout << "\n=== Launching interactive viewer ===" << std::endl;
    std::cout << "Available keyboard shortcuts:" << std::endl;
    std::cout << "  - [N] / [Right Arrow]: Next image" << std::endl;
    std::cout << "  - [P] / [Left Arrow]: Previous image" << std::endl;
    std::cout << "  - [S]: Sobel detector" << std::endl;
    std::cout << "  - [L]: Laplacian detector" << std::endl;
    std::cout << "  - [C]: Canny detector" << std::endl;
    std::cout << "  - [O]: Auto-optimize parameters for current image" << std::endl;
    std::cout << "  - [Q] or [ESC]: Quit\n" << std::endl;

    while (true)
    {
        if (state.needsRedraw)
        {
            renderViewer(state);
            state.needsRedraw = false;
        }

        const int key = cv::waitKey(30);

        if (key == 27 || key == 'q' || key == 'Q') { break; }
        if (key == 'n' || key == 'N' || key == 2555904 || key == 65363) // N key or Right Arrow
        {
            if (state.trackbarImage + 1 < static_cast<int>(images.size()))
            {
                state.trackbarImage++;
                cv::setTrackbarPos("Image", state.winName, state.trackbarImage);
                state.optimizeCurrent();
            }
        }
        else if (key == 'p' || key == 'P' || key == 2424832 || key == 65361) // P key or Left Arrow
        {
            if (state.trackbarImage > 0)
            {
                state.trackbarImage--;
                cv::setTrackbarPos("Image", state.winName, state.trackbarImage);
                state.optimizeCurrent();
            }
        }
        else if (key == 's' || key == 'S')
        {
            state.trackbarDetector = DETECTOR_SOBEL;
            cv::setTrackbarPos("Detector", state.winName, state.trackbarDetector);
            state.optimizeCurrent();
        }
        else if (key == 'l' || key == 'L')
        {
            state.trackbarDetector = DETECTOR_LAPLACIAN;
            cv::setTrackbarPos("Detector", state.winName, state.trackbarDetector);
            state.optimizeCurrent();
        }
        else if (key == 'c' || key == 'C')
        {
            state.trackbarDetector = DETECTOR_CANNY;
            cv::setTrackbarPos("Detector", state.winName, state.trackbarDetector);
            state.optimizeCurrent();
        }
        else if (key == 'o' || key == 'O') { state.optimizeCurrent(); }
    }

    cv::destroyWindow(state.winName);
}
} // namespace Interactive

#endif // OPENCV_INTERACTIVE_H
