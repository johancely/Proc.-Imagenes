#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class FrameDifferencer
{
public:
    FrameDifferencer(int threshold_val, int min_contour_area, double learning_rate = 0.0);

    void setBackground(const cv::Mat& frame);
    cv::Mat process(const cv::Mat& frame);
    std::vector<cv::Rect> getRegions(const cv::Mat& mask);
    void updateBackground(const cv::Mat& frame);

private:
    cv::Mat background_;
    int threshold_val_;
    int min_area_;
    double learning_rate_;
    cv::Mat kernel_;
};
