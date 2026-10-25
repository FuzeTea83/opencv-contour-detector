#ifndef OPENCV_SOBEL_H
#define OPENCV_SOBEL_H

#include <algorithm>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

#include "stats.h"

namespace Sobel
{

struct Params
{
    int ksize = 3;
    double scale = 1.0;
    double delta = 0.0;
    int threshold = 128;
    int blurSize = 5;

    void toString() const
    {
        std::cout << "Sobel Params: ksize=" << ksize
                  << ", scale=" << scale
                  << ", delta=" << delta
                  << ", threshold=" << threshold
                  << ", blurSize=" << blurSize
                  << std::endl;
    }

    static Params average(const Params *params, const size_t count)
    {
        if (count == 0) return Params{};
        Params avg{};
        double sumK = 0, sumScale = 0, sumDelta = 0, sumThresh = 0, sumBlur = 0;
        for (size_t i = 0; i < count; ++i)
        {
            sumK += params[i].ksize;
            sumScale += params[i].scale;
            sumDelta += params[i].delta;
            sumThresh += params[i].threshold;
            sumBlur += params[i].blurSize;
        }
        avg.ksize = std::max(1, static_cast<int>(sumK / count));
        if (avg.ksize % 2 == 0) avg.ksize += 1;
        avg.scale = sumScale / count;
        avg.delta = sumDelta / count;
        avg.threshold = static_cast<int>(sumThresh / count);
        avg.blurSize = std::max(0, static_cast<int>(sumBlur / count));
        if (avg.blurSize > 0 && avg.blurSize % 2 == 0) avg.blurSize += 1;
        return avg;
    }
};

/**
 * @brief Computes the normalized magnitude map of the Sobel gradient [0, 255].
 */
static cv::Mat computeSobelMagnitude(const cv::Mat &image, const Params &params, int ddepth = CV_16S)
{
    cv::Mat src;
    if (params.blurSize > 0)
    {
        cv::GaussianBlur(image, src, cv::Size(params.blurSize, params.blurSize), 0);
    }
    else
    {
        src = image;
    }

    cv::Mat gradX, gradY;
    cv::Sobel(src, gradX, ddepth, 1, 0, params.ksize, params.scale, params.delta, cv::BORDER_DEFAULT);
    cv::Sobel(src, gradY, ddepth, 0, 1, params.ksize, params.scale, params.delta, cv::BORDER_DEFAULT);

    cv::Mat absGradX, absGradY;
    cv::convertScaleAbs(gradX, absGradX);
    cv::convertScaleAbs(gradY, absGradY);

    cv::Mat sobel;
    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, sobel);

    double sobMin, sobMax;
    cv::minMaxLoc(sobel, &sobMin, &sobMax);
    if (sobMax <= 1e-5)
    {
        return cv::Mat::zeros(sobel.size(), CV_8U);
    }

    cv::Mat normalized;
    sobel.convertTo(normalized, CV_8U, 255.0 / sobMax, 0);
    return normalized;
}

/**
 * @brief Binarizes the normalized magnitude image: edges = 0 (black), background = 255 (white).
 */
static cv::Mat thresholdContour(const cv::Mat &normalizedSobel, const int threshold)
{
    cv::Mat result;
    cv::threshold(normalizedSobel, result, threshold, 255, cv::THRESH_BINARY_INV);
    return result;
}

/**
 * @brief Complete edge detection using Sobel.
 */
static cv::Mat contour(const cv::Mat &image, const Params &params, const int ddepth = CV_16S)
{
    const cv::Mat normalized = computeSobelMagnitude(image, params, ddepth);
    return thresholdContour(normalized, params.threshold);
}

static Stats evaluateParams(const cv::Mat &image, const cv::Mat &ref, const Params &params)
{
    const cv::Mat result = contour(image, params);
    return calculateStats(result, ref);
}

/**
 * @brief Optimized search for the best hyperparameters for Sobel (blurSize, ksize, threshold).
 * Optimization: The gradient is calculated only once per (blur, ksize) combination.
 */
static std::pair<Params, Stats> findBestParams(const cv::Mat &image, const cv::Mat &ref)
{
    ContourEvaluator evaluator(ref);
    Params bestParams;
    Stats bestStats;
    float bestFScore = -1.0f;

    for (const int blur : {0, 3, 5, 7})
    {
        for (const int ksize : {1, 3, 5})
        {
            Params p;
            p.blurSize = blur;
            p.ksize = ksize;

            const cv::Mat normalized = computeSobelMagnitude(image, p);

            // Coarse search for the threshold
            int localBestThresh = 50;
            float localBestScore = -1.0f;
            Stats localBestStats;

            for (int t = 10; t <= 240; t += 10)
            {
                const cv::Mat binarized = thresholdContour(normalized, t);
                const Stats s = evaluator.evaluate(binarized);
                if (s.F1 > localBestScore)
                {
                    localBestScore = s.F1;
                    localBestThresh = t;
                    localBestStats = s;
                }
            }

            // Fine tuning around the best threshold
            const int tMin = std::max(1, localBestThresh - 12);
            const int tMax = std::min(254, localBestThresh + 12);
            for (int t = tMin; t <= tMax; t += 2)
            {
                const cv::Mat binarized = thresholdContour(normalized, t);
                const Stats s = evaluator.evaluate(binarized);
                if (s.F1 > localBestScore)
                {
                    localBestScore = s.F1;
                    localBestThresh = t;
                    localBestStats = s;
                }
            }

            if (localBestScore > bestFScore)
            {
                bestFScore = localBestScore;
                p.threshold = localBestThresh;
                bestParams = p;
                bestStats = localBestStats;
            }
        }
    }

    return {bestParams, bestStats};
}

} // namespace Sobel

#endif //OPENCV_SOBEL_H
