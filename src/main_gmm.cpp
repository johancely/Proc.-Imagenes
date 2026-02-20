#include "gmm_segmenter.h"

#include <iostream>

int main(int argc, char** argv)
{
    cv::VideoCapture cap;
    if (argc > 1)
    {
        cap.open(argv[1]);
    }
    else
    {
        cap.open(0);
    }

    if (!cap.isOpened())
    {
        std::cerr << "No se pudo abrir la fuente de video." << std::endl;
        return 1;
    }

    GMMSegmenter segmenter;

    cv::Mat frame;
    while (cap.read(frame))
    {
        cv::Mat mask = segmenter.apply(frame);
        std::vector<cv::Rect> regions = segmenter.getRegions(mask);

        cv::Mat vis = frame.clone();
        for (const auto& rect : regions)
        {
            cv::rectangle(vis, rect, cv::Scalar(0, 255, 0), 2);
        }

        cv::Mat background = segmenter.getBackground();

        cv::imshow("GMM - Original", vis);
        cv::imshow("GMM - Mascara", mask);
        if (!background.empty())
        {
            cv::imshow("GMM - Fondo", background);
        }

        if ((cv::waitKey(30) & 0xFF) == 27)
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
