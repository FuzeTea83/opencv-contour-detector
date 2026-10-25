#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "canny.h"
#include "interactive.h"
#include "laplacian.h"
#include "sobel.h"
#include "stats.h"

static void getImages(const std::filesystem::path &path, const std::filesystem::path &refPath,
                      std::vector<cv::Mat> &images, std::vector<cv::Mat> &references,
                      std::vector<std::string> &filenames)
{
    if (!std::filesystem::exists(path))
    {
        std::cerr << "Image path does not exist: " << path << std::endl;
        return;
    }

    for (const auto &entry: std::filesystem::directory_iterator(path))
    {
        if (!entry.is_regular_file()) { continue; }

        std::string filename = entry.path().stem().string();
        std::filesystem::path referencePath = refPath / (filename + "_gt_binary.pgm");
        if (!std::filesystem::exists(referencePath))
        {
            std::cerr << "Reference image not found: " << referencePath << std::endl;
            continue;
        }

        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        cv::Mat reference = cv::imread(referencePath.string(), cv::IMREAD_GRAYSCALE);

        if (image.empty())
        {
            std::cerr << "Could not open image: " << entry.path() << std::endl;
            continue;
        }

        if (reference.empty())
        {
            std::cerr << "Could not open reference image: " << referencePath << std::endl;
            continue;
        }

        images.push_back(image);
        references.push_back(reference);
        filenames.push_back(filename);
    }
}

static void showGraph(const std::vector<Stats> &statsList, const std::string &detectorName)
{
    if (statsList.empty()) return;

    constexpr int widthScale = 40;
    constexpr int heightScale = 400;
    const int count = static_cast<int>(statsList.size());
    cv::Mat graph(heightScale, std::max(count * widthScale, 200), CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 0; i < count; ++i)
    {
        const int x = i * widthScale + widthScale / 2;
        const int yP   = std::clamp(heightScale - static_cast<int>(statsList[i].P * heightScale), 0, heightScale - 1);
        const int yFPR = std::clamp(heightScale - static_cast<int>(statsList[i].FPR * heightScale), 0, heightScale - 1);
        const int yFNR = std::clamp(heightScale - static_cast<int>(statsList[i].FNR * heightScale), 0, heightScale - 1);
        const int yF1  = std::clamp(heightScale - static_cast<int>(statsList[i].F1 * heightScale), 0, heightScale - 1);

        cv::drawMarker(graph, cv::Point(x, yP),   cv::Scalar(0, 0, 255),   cv::MARKER_CROSS, 16, 2); // Red = P
        cv::drawMarker(graph, cv::Point(x, yFPR), cv::Scalar(0, 180, 0),   cv::MARKER_TILTED_CROSS, 12, 2); // Green = FPR
        cv::drawMarker(graph, cv::Point(x, yFNR), cv::Scalar(255, 0, 0),   cv::MARKER_STAR, 12, 2); // Blue = FNR
        cv::drawMarker(graph, cv::Point(x, yF1),  cv::Scalar(200, 0, 200), cv::MARKER_DIAMOND, 12, 2); // Magenta = F1
    }

    const std::string winName = "Graph - " + detectorName;
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    cv::imshow(winName, graph);

    const float avgP   = std::accumulate(statsList.begin(), statsList.end(), 0.0f, [](const float sum, const Stats &s) { return sum + s.P; }) / static_cast<float>(count);
    const float avgFPR = std::accumulate(statsList.begin(), statsList.end(), 0.0f, [](const float sum, const Stats &s) { return sum + s.FPR; }) / static_cast<float>(count);
    const float avgFNR = std::accumulate(statsList.begin(), statsList.end(), 0.0f, [](const float sum, const Stats &s) { return sum + s.FNR; }) / static_cast<float>(count);
    const float avgF   = std::accumulate(statsList.begin(), statsList.end(), 0.0f, [](const float sum, const Stats &s) { return sum + s.F1; }) / static_cast<float>(count);

    std::cout << "\n=== Average Statistics [" << detectorName << "] ===" << std::endl;
    std::cout << "  Average Performance (P) : " << avgP << std::endl;
    std::cout << "  Average FPR (False Positive Rate): " << avgFPR << std::endl;
    std::cout << "  Average FNR (False Negative Rate): " << avgFNR << std::endl;
    std::cout << "  Average F-Score (F1)           : " << avgF << std::endl;
}

namespace {
template <typename ParamsType>
struct DetectorResult
{
    size_t index = 0;
    std::string name;
    ParamsType params;
    Stats stats;
    cv::Mat resultImage;
};
}

template <typename ParamsType, typename FindFunc, typename ContourFunc>
static void runDetector(const std::string &detectorName,
                        const std::vector<cv::Mat> &images,
                        const std::vector<cv::Mat> &references,
                        const std::vector<std::string> &filenames,
                        FindFunc findBest,
                        ContourFunc contourFunc)
{
    std::cout << "\n--- Optimization and evaluation: " << detectorName << " ---" << std::endl;

    std::vector<DetectorResult<ParamsType>> results(images.size());
    std::vector<Stats> statsList(images.size());
    std::vector<ParamsType> paramsList(images.size());

    // Parallel computation over images for optimal performance
    cv::parallel_for_(cv::Range(0, static_cast<int>(images.size())), [&](const cv::Range &range)
    {
        for (int i = range.start; i < range.end; ++i)
        {
            const auto [params, stats] = findBest(images[i], references[i]);
            const cv::Mat res = contourFunc(images[i], params);

            results[i] = {static_cast<size_t>(i), filenames[i], params, stats, res};
            statsList[i] = stats;
            paramsList[i] = params;
        }
    });

    size_t bestIdx = 0;
    float maxF = -1.0f;

    for (size_t i = 0; i < images.size(); ++i)
    {
        if (statsList[i].F1 > maxF)
        {
            maxF = statsList[i].F1;
            bestIdx = i;
        }

        std::cout << "[" << detectorName << "] " << filenames[i] << " -> ";
        paramsList[i].toString();
        std::cout << "       ";
        statsList[i].toString();
    }

    const ParamsType avgParams = ParamsType::average(paramsList.data(), paramsList.size());
    std::cout << "\n[Average Parameters - " << detectorName << "]: ";
    avgParams.toString();

    showGraph(statsList, detectorName);

    std::cout << "\nBest image according to F1: " << filenames[bestIdx] << " (F1 = " << maxF << ")\n" << std::endl;
}

int main(const int argc, char **argv)
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path> <reference_path> [sobel|laplacian|canny|all|interactive]" << std::endl;
        return 1;
    }

    const std::filesystem::path path = argv[1];
    const std::filesystem::path refPath = argv[2];
    const std::string mode = (argc >= 4) ? argv[3] : "all";

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> references;
    std::vector<std::string> filenames;

    getImages(path, refPath, images, references, filenames);
    if (images.empty())
    {
        std::cerr << "No images found in: " << path << std::endl;
        return 1;
    }

    std::cout << "Number of loaded images: " << images.size() << std::endl;

    if (mode == "interactive" || mode == "gui" || mode == "-i" || mode == "--interactive")
    {
        int initialDetector = Interactive::DETECTOR_CANNY;
        if (argc >= 5)
        {
            const std::string det = argv[4];
            if (det == "sobel") initialDetector = Interactive::DETECTOR_SOBEL;
            else if (det == "laplacian") initialDetector = Interactive::DETECTOR_LAPLACIAN;
            else if (det == "canny") initialDetector = Interactive::DETECTOR_CANNY;
        }

        // Launch the interactive viewer with trackbars only when explicitly requested
        Interactive::run(images, references, filenames, initialDetector);
        return 0;
    }

    if (mode == "sobel" || mode == "all")
    {
        runDetector<Sobel::Params>("Sobel", images, references, filenames,
                                   Sobel::findBestParams,
                                   [](const cv::Mat &img, const Sobel::Params &p) { return Sobel::contour(img, p); });
    }

    if (mode == "laplacian" || mode == "all")
    {
        runDetector<Laplacian::Params>("Laplacian", images, references, filenames,
                                       Laplacian::findBestParams,
                                       [](const cv::Mat &img, const Laplacian::Params &p) { return Laplacian::contour(img, p); });
    }

    if (mode == "canny" || mode == "all")
    {
        runDetector<Canny::Params>("Canny", images, references, filenames,
                                   Canny::findBestParams,
                                   [](const cv::Mat &img, const Canny::Params &p) { return Canny::contour(img, p); });
    }

    if (mode == "sobel" || mode == "laplacian" || mode == "canny" || mode == "all")
    {
        std::cout << "Pres a key on a window to quit..." << std::endl;
        cv::waitKey(0);
    }
    else
    {
        std::cerr << "Unknown mode: " << mode << " (valid: sobel, laplacian, canny, all, interactive)" << std::endl;
        return 1;
    }

    return 0;
}
