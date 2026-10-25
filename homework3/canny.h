#ifndef OPENCV_CANNY_H
#define OPENCV_CANNY_H

#include <algorithm>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

#include "stats.h"

namespace Canny
{

struct Params
{
    int threshold1 = 50;
    int threshold2 = 100;
    int apertureSize = 3;
    int blurSize = 3;
    bool L2gradient = false;

    void toString() const
    {
        std::cout << "Canny Params: threshold1=" << threshold1
                  << ", threshold2=" << threshold2
                  << ", apertureSize=" << apertureSize
                  << ", blurSize=" << blurSize
                  << ", L2gradient=" << (L2gradient ? "true" : "false")
                  << std::endl;
    }

    static Params average(const Params *params, const size_t count)
    {
        if (count == 0) return Params{};
        Params avg{};
        double sumT1 = 0, sumT2 = 0, sumAp = 0, sumBlur = 0;
        int l2Count = 0;

        for (size_t i = 0; i < count; ++i)
        {
            sumT1 += params[i].threshold1;
            sumT2 += params[i].threshold2;
            sumAp += params[i].apertureSize;
            sumBlur += params[i].blurSize;
            if (params[i].L2gradient) l2Count++;
        }

        avg.threshold1 = static_cast<int>(sumT1 / count);
        avg.threshold2 = static_cast<int>(sumT2 / count);
        avg.apertureSize = std::max(3, static_cast<int>(sumAp / count));
        if (avg.apertureSize % 2 == 0) avg.apertureSize += 1;
        avg.blurSize = std::max(1, static_cast<int>(sumBlur / count));
        if (avg.blurSize % 2 == 0) avg.blurSize += 1;
        avg.L2gradient = (l2Count * 2 >= static_cast<int>(count));

        return avg;
    }
};

/**
 * @brief Edge detection using Canny.
 * Returns an inverted binary image: edges = 0 (black), background = 255 (white).
 */
static cv::Mat contour(const cv::Mat &image, const Params &params)
{
    cv::Mat blurred;
    if (params.blurSize > 1)
    {
        cv::GaussianBlur(image, blurred, cv::Size(params.blurSize, params.blurSize), 0);
    }
    else
    {
        blurred = image;
    }

    cv::Mat canny;
    cv::Canny(blurred, canny, params.threshold1, params.threshold2, params.apertureSize, params.L2gradient);

    cv::Mat inverted;
    cv::bitwise_not(canny, inverted);
    return inverted;
}

static Stats evaluateParams(const cv::Mat &image, const cv::Mat &ref, const Params &params)
{
    const cv::Mat result = contour(image, params);
    return calculateStats(result, ref);
}

/**
 * @brief Estimates a basic threshold using Otsu's method on the Sobel gradient.
 */
static int estimateOtsuThreshold(const cv::Mat &gray)
{
    cv::Mat gx, gy, mag;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);

    cv::Mat mag8u;
    cv::normalize(mag, mag8u, 0, 255, cv::NORM_MINMAX);
    mag8u.convertTo(mag8u, CV_8U);

    return static_cast<int>(cv::threshold(mag8u, mag8u, 0, 255, cv::THRESH_OTSU));
}

/**
 * @brief Optimized search for the best hyperparameters for Canny.
 * Fast multi-scale search with adaptive hysteresis thresholding and fine tuning.
 */
static std::pair<Params, Stats> findBestParams(const cv::Mat &image, const cv::Mat &ref)
{
    ContourEvaluator evaluator(ref);
    Params bestParams;
    Stats bestStats;
    float bestFScore = -1.0f;

    for (const int blur : {1, 3, 5, 7})
    {
        cv::Mat blurred;
        if (blur > 1)
        {
            cv::GaussianBlur(image, blurred, cv::Size(blur, blur), 0);
        }
        else
        {
            blurred = image;
        }

        const int otsuT = estimateOtsuThreshold(blurred);

        // Restricted and relevant set of thresholds based on Otsu and exploration
        std::vector<int> candidateT1;
        for (int t = 20; t <= 160; t += 25)
        {
            candidateT1.push_back(t);
        }
        if (otsuT > 0)
        {
            candidateT1.push_back(otsuT);
            candidateT1.push_back(std::max(10, otsuT / 2));
        }

        for (const int aperture : {3})
        {
            for (const bool l2 : {false, true})
            {
                for (const int t1 : candidateT1)
                {
                    for (const float ratio : {1.5f, 2.0f, 2.5f, 3.0f})
                    {
                        const int t2 = std::min(255, static_cast<int>(t1 * ratio));
                        if (t2 <= t1) continue;

                        Params p{.threshold1 = t1, .threshold2 = t2, .apertureSize = aperture, .blurSize = blur, .L2gradient = l2};
                        cv::Mat canny;
                        cv::Canny(blurred, canny, t1, t2, aperture, l2);

                        cv::Mat result;
                        cv::bitwise_not(canny, result);

                        const Stats s = evaluator.evaluate(result);
                        if (s.F1 > bestFScore)
                        {
                            bestFScore = s.F1;
                            bestParams = p;
                            bestStats = s;
                        }
                    }
                }
            }
        }
    }

    // Fine tuning around the best found t1 / t2
    if (bestFScore > 0.0f)
    {
        cv::Mat blurred;
        if (bestParams.blurSize > 1)
        {
            cv::GaussianBlur(image, blurred, cv::Size(bestParams.blurSize, bestParams.blurSize), 0);
        }
        else
        {
            blurred = image;
        }

        const int baseT1 = bestParams.threshold1;
        const int baseT2 = bestParams.threshold2;

        for (int dt1 = -10; dt1 <= 10; dt1 += 5)
        {
            const int t1 = std::clamp(baseT1 + dt1, 5, 240);
            for (int dt2 = -15; dt2 <= 15; dt2 += 5)
            {
                const int t2 = std::clamp(baseT2 + dt2, t1 + 5, 255);

                cv::Mat canny;
                cv::Canny(blurred, canny, t1, t2, bestParams.apertureSize, bestParams.L2gradient);

                cv::Mat result;
                cv::bitwise_not(canny, result);

                const Stats s = evaluator.evaluate(result);
                if (s.F1 > bestFScore)
                {
                    bestFScore = s.F1;
                    bestParams.threshold1 = t1;
                    bestParams.threshold2 = t2;
                    bestStats = s;
                }
            }
        }
    }

    return {bestParams, bestStats};
}

} // namespace Canny

#endif //OPENCV_CANNY_H
