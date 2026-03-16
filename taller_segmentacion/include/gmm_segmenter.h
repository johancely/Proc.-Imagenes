#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <vector>

class GMMSegmenter
{
public:
    struct Config
    {
        int history = 500;
        double var_threshold = 16.0;
        bool detect_shadows = true;
        int min_area = 1500;
        int morph_open_k = 3;
        int morph_close_k = 7;
    };

    explicit GMMSegmenter(const Config& cfg = Config{});

    cv::Mat apply(const cv::Mat& frame);
    std::vector<cv::Rect> getRegions(const cv::Mat& mask) const;
    cv::Mat getBackground() const;

private:
    Config cfg_;
    cv::Ptr<cv::BackgroundSubtractorMOG2> pMOG2_;
    cv::Mat kernel_open_;
    cv::Mat kernel_close_;
};
