#include "frame_diff.h"

#include <iostream>
#include <string>

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

    FrameDifferencer differencer(30, 1500, 0.0);

    cv::Mat frame;
    if (!cap.read(frame))
    {
        std::cerr << "No se pudo leer el primer cuadro." << std::endl;
        return 1;
    }

    differencer.setBackground(frame);

    while (cap.read(frame))
    {
        cv::Mat mask = differencer.process(frame);
        std::vector<cv::Rect> regions = differencer.getRegions(mask);

        cv::Mat vis = frame.clone();
        for (const auto& rect : regions)
        {
            cv::rectangle(vis, rect, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Frame Differencing - Original", vis);
        cv::imshow("Frame Differencing - Mascara", mask);

        if ((cv::waitKey(30) & 0xFF) == 27)
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
