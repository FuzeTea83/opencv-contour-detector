#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>

static cv::Mat drawHistogram(const cv::Mat &gray)
{
    constexpr int histSize = 256;
    float range[] = { 0, 256 };
    const float *histRange = { range };

    cv::Mat hist;
    cv::calcHist(&gray, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange);

    // Normalize the histogram to fit in the display image
    double minVal = 0;
    double maxVal = 0;
    cv::minMaxLoc(hist, &minVal, &maxVal);

    cv::Mat histImage(histSize, histSize, CV_8UC1, cv::Scalar(255));

    constexpr int htp = static_cast<int>(histSize * 0.9);

    for (int h = 0; h < histSize; ++h)
    {
        const float binVal = hist.at<float>(h);
        if (binVal <= 0) continue;

        const int intensity = static_cast<int>(binVal * htp / maxVal);
        cv::line(histImage, cv::Point(h, histSize - 1), cv::Point(h, histSize - intensity - 1), cv::Scalar(0));
    }

    return histImage;
}

int main(const int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    // Load the image
    const char* imagePath = argc > 1 ? argv[1] : "AI_totoro.png";

    if (!std::filesystem::exists(imagePath)) {
        std::cerr << "Image file does not exist: " << imagePath << std::endl;
        return -1;
    }

    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Could not open or find the image: " << imagePath << std::endl;
        return -1;
    }

    // Equalize the histogram of the image
    cv::Mat dst;
    cv::equalizeHist(image, dst);

    // Draw histograms for the original and equalized images
    const cv::Mat h1 = drawHistogram(image);
    const cv::Mat h2 = drawHistogram(dst);

    // Create windows to display the images
    cv::namedWindow("1. Original", cv::WINDOW_NORMAL);
    cv::namedWindow("2. Expanded", cv::WINDOW_NORMAL);
    cv::namedWindow("3. Histogram original", cv::WINDOW_NORMAL);
    cv::namedWindow("4. Histogram expanded", cv::WINDOW_NORMAL);
    cv::startWindowThread();
    cv::imshow("1. Original", image);
    cv::imshow("2. Expanded", dst);
    cv::imshow("3. Histogram original", h1);
    cv::imshow("4. Histogram expanded", h2);

    // Press Escape
    while (true) {
        if ((cv::waitKey(50) & 0xFF) == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}