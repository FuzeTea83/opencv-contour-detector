#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>

static void onMouse(int event, int x, int y, int flags, void* params);
static cv::Mat drawProfileImage(const cv::Mat& src, int y);

int main(const int argc, char** argv)
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    // Load the image
    const char* imagePath = argc > 1 ? argv[1] : "tower.jpg";

    if (!std::filesystem::exists(imagePath)) {
        std::cerr << "Image file does not exist: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Could not open or find the image: " << imagePath << std::endl;
        return -1;
    }

    // Convert the grayscale image to BGR format to display the profile line in color
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);


    // Create a white image for the profile
    const auto profileImage = cv::Mat(image.rows, image.cols, CV_8UC3, cv::Scalar(255, 255, 255));


    // Create windows to display the images
    cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    cv::namedWindow("Profile Image", cv::WINDOW_NORMAL);
    cv::startWindowThread();
    cv::imshow("Original Image", image);
    cv::imshow("Profile Image", profileImage);

    // Set the mouse callback function
    cv::setMouseCallback("Original Image", onMouse, &image);

    // Press Escape
    while (true) {
        if ((cv::waitKey(50) & 0xFF) == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}

static void onMouse(const int event, const int x, const int y, int flags, void* params)
{
    const auto src = static_cast<cv::Mat *>(params);
    auto copy = src->clone();

    // Draw a horizontal line at the clicked position and show the profile image
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        cv::line(copy, cv::Point(0, y), cv::Point(copy.cols, y), cv::Scalar(0, 0, 255), 1);
        const cv::Mat profileImage = drawProfileImage(*src, y);
        cv::imshow("Original Image", copy);
        cv::imshow("Profile Image", profileImage);
    }
}


static cv::Mat drawProfileImage(const cv::Mat& src, int y)
{
    auto profileImage = cv::Mat(src.rows, 255, CV_8UC3, cv::Scalar(255, 255, 255));

    const auto *col = src.ptr<cv::Vec3b>(y);

    auto value = col[0];
    auto average = (value[0] + value[1] + value[2]) / 3;
    int lastY = average;

    for (auto x = 1; x < src.cols; x++)
    {
        value = col[x];
        average = (value[0] + value[1] + value[2]) / 3;

        cv::line(profileImage, cv::Point(x - 1, 255 - lastY), cv::Point(x, 255 - average), cv::Scalar(0), 1);
        lastY = average;
    }

    return profileImage;
}
