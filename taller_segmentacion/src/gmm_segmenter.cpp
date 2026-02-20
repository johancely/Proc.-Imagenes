#include "gmm_segmenter.h"

namespace
{
int normalizeKernelSize(int value)
{
    int size = std::max(1, value);
    if (size % 2 == 0)
    {
        ++size;
    }
    return size;
}
}  // namespace

GMMSegmenter::GMMSegmenter(const Config& cfg) : cfg_(cfg)
{
    pMOG2_ = cv::createBackgroundSubtractorMOG2(cfg_.history, cfg_.var_threshold, cfg_.detect_shadows);

    const int open_k = normalizeKernelSize(cfg_.morph_open_k);
    const int close_k = normalizeKernelSize(cfg_.morph_close_k);
    kernel_open_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(open_k, open_k));
    kernel_close_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(close_k, close_k));
}

cv::Mat GMMSegmenter::apply(const cv::Mat& frame)
{
    if (frame.empty())
    {
        return {};
    }

    cv::Mat fg_mask;
    pMOG2_->apply(frame, fg_mask);

    cv::Mat clean_mask;
    cv::threshold(fg_mask, clean_mask, 200, 255, cv::THRESH_BINARY);
    cv::morphologyEx(clean_mask, clean_mask, cv::MORPH_OPEN, kernel_open_);
    cv::morphologyEx(clean_mask, clean_mask, cv::MORPH_CLOSE, kernel_close_);

    return clean_mask;
}

std::vector<cv::Rect> GMMSegmenter::getRegions(const cv::Mat& mask) const
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat mask_copy = mask.clone();
    cv::findContours(mask_copy, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> regions;
    for (const auto& contour : contours)
    {
        if (cv::contourArea(contour) >= cfg_.min_area)
        {
            regions.push_back(cv::boundingRect(contour));
        }
    }

    return regions;
}

cv::Mat GMMSegmenter::getBackground() const
{
    cv::Mat background;
    pMOG2_->getBackgroundImage(background);
    return background;
}
