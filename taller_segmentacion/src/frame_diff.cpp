#include "frame_diff.h"

#include <algorithm>

FrameDifferencer::FrameDifferencer(int threshold_val, int min_contour_area, double learning_rate)
    : threshold_val_(threshold_val),
      min_area_(min_contour_area),
      learning_rate_(std::clamp(learning_rate, 0.0, 1.0)),
      kernel_(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)))
{
}

void FrameDifferencer::setBackground(const cv::Mat& frame)
{
    if (frame.empty())
    {
        return;
    }

    cv::cvtColor(frame, background_, cv::COLOR_BGR2GRAY);
}

cv::Mat FrameDifferencer::process(const cv::Mat& frame)
{
    if (frame.empty())
    {
        return {};
    }

    if (background_.empty())
    {
        setBackground(frame);
        return cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
    }

    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

    cv::Mat diff_frame;
    cv::absdiff(background_, gray_frame, diff_frame);

    cv::Mat binary_mask;
    cv::threshold(diff_frame, binary_mask, threshold_val_, 255, cv::THRESH_BINARY);

    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_OPEN, kernel_);
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_CLOSE, kernel_);

    updateBackground(frame);

    return binary_mask;
}

std::vector<cv::Rect> FrameDifferencer::getRegions(const cv::Mat& mask)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat mask_copy = mask.clone();
    cv::findContours(mask_copy, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> regions;
    for (const auto& contour : contours)
    {
        if (cv::contourArea(contour) >= min_area_)
        {
            regions.push_back(cv::boundingRect(contour));
        }
    }

    return regions;
}

void FrameDifferencer::updateBackground(const cv::Mat& frame)
{
    if (learning_rate_ <= 0.0 || frame.empty())
    {
        return;
    }

    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

    if (background_.empty())
    {
        background_ = gray_frame.clone();
        return;
    }

    cv::addWeighted(background_, learning_rate_, gray_frame, 1.0 - learning_rate_, 0.0, background_);
}
