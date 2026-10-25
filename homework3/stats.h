#ifndef OPENCV_STATS_H
#define OPENCV_STATS_H

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include <opencv2/core/mat.hpp>

struct Stats
{
    float P = 0.0f;   // Performance
    float FPR = 0.0f; // False Positive Rate
    float FNR = 0.0f; // False Negative Rate
    float F1 = 0.0f;   // F-score

    void toString() const
    {
        std::cout << std::fixed << std::setprecision(4)
                  << "Stats: P=" << P
                  << ", FPR=" << FPR
                  << ", FNR=" << FNR
                  << ", F1=" << F1
                  << std::endl;
    }
};

/**
 * @brief Optimized evaluation structure with epoch-based tagging
 * to avoid repeated memory allocations and memsets.
 */
class ContourEvaluator
{
public:
    explicit ContourEvaluator(const cv::Mat &ref)
        : m_ref(ref), m_rows(ref.rows), m_cols(ref.cols), m_marked(ref.rows * ref.cols, 0)
    {
        CV_Assert(!ref.empty() && ref.type() == CV_8UC1);
        for (int y = 0; y < m_rows; ++y)
        {
            const auto *ptr = ref.ptr<uchar>(y);
            for (int x = 0; x < m_cols; ++x)
            {
                if (ptr[x] == 0)
                {
                    m_refContours++;
                }
            }
        }
    }

    Stats evaluate(const cv::Mat &src)
    {
        CV_Assert(src.rows == m_rows && src.cols == m_cols && src.type() == CV_8UC1);

        m_epoch++;
        if (m_epoch == 0)
        {
            std::ranges::fill(m_marked, 0);
            m_epoch = 1;
        }

        const uint32_t currentEpoch = m_epoch;
        int foundContours = 0;
        int correctContours = 0;

        const int rows = m_rows;
        const int cols = m_cols;
        const uchar *refData = m_ref.data;
        const size_t refStep = m_ref.step;
        const uchar *srcData = src.data;
        const size_t srcStep = src.step;
        uint32_t *markData = m_marked.data();

        for (int y = 0; y < rows; ++y)
        {
            const uchar *ptrSrc = srcData + y * srcStep;
            const int yMin = (y > 0) ? (y - 1) : 0;
            const int yMax = (y + 1 < rows) ? (y + 1) : (rows - 1);

            for (int x = 0; x < cols; ++x)
            {
                if (ptrSrc[x] != 0) continue;
                foundContours++;

                const int xMin = (x > 0) ? (x - 1) : 0;
                const int xMax = (x + 1 < cols) ? (x + 1) : (cols - 1);

                bool foundInRef = false;
                for (int ny = yMin; ny <= yMax && !foundInRef; ++ny)
                {
                    const uchar *ptrRefRow = refData + ny * refStep;
                    uint32_t *markRow = markData + ny * cols;

                    for (int nx = xMin; nx <= xMax; ++nx)
                    {
                        if (ptrRefRow[nx] == 0 && markRow[nx] != currentEpoch)
                        {
                            correctContours++;
                            markRow[nx] = currentEpoch;
                            foundInRef = true;
                            break;
                        }
                    }
                }
            }
        }

        const int falsePositive = foundContours - correctContours;
        const int falseNegative = m_refContours - correctContours;
        const int total         = correctContours + falsePositive + falseNegative;

        Stats stats;
        if (total == 0) return stats;

        stats.P   = static_cast<float>(correctContours) / static_cast<float>(total);
        stats.FPR = static_cast<float>(falsePositive)   / static_cast<float>(total);
        stats.FNR = static_cast<float>(falseNegative)   / static_cast<float>(total);

        const float precision = foundContours > 0
            ? static_cast<float>(correctContours) / static_cast<float>(foundContours) : 0.0f;
        const float recall = m_refContours > 0
            ? static_cast<float>(correctContours) / static_cast<float>(m_refContours) : 0.0f;

        stats.F1 = (precision + recall) > 0.0f
            ? (2.0f * precision * recall) / (precision + recall) : 0.0f;

        return stats;
    }

private:
    cv::Mat m_ref;
    int m_rows = 0;
    int m_cols = 0;
    int m_refContours = 0;
    uint32_t m_epoch = 1;
    std::vector<uint32_t> m_marked;
};

/**
 * @brief Computes evaluation statistics between detection and ground truth.
 * Convention: edges are represented by black pixels (value == 0).
 */
static Stats calculateStats(const cv::Mat &src, const cv::Mat &ref)
{
    ContourEvaluator evaluator(ref);
    return evaluator.evaluate(src);
}

#endif //OPENCV_STATS_H