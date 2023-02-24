#pragma once

#include <vector>

#include <opencv2/highgui/highgui.hpp>


// //loguru
// #include <loguru.hpp>

namespace radu{
namespace utils{



//convert an OpenCV type to a string value
inline std::string type2string(int type) {
    std::string r;

    unsigned char depth = type & CV_MAT_DEPTH_MASK;
    unsigned char chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}


//return the byteDepth of this cv mat. return one of the following  CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F
inline unsigned char type2byteDepth(int type) {
    unsigned char depth = type & CV_MAT_DEPTH_MASK;

    return depth;
}

// Removed them because now we support rgb texture in EasyGL and noy only 4 channel ones
// inline void create_alpha_mat(const cv::Mat& mat, cv::Mat_<cv::Vec4b>& dst){
//     std::vector<cv::Mat> matChannels;
//     cv::split(mat, matChannels);

//     cv::Mat alpha=cv::Mat(mat.rows,mat.cols, CV_8UC1);
//     alpha.setTo(cv::Scalar(255));
//     matChannels.push_back(alpha);

//     cv::merge(matChannels, dst);
// }

template <class T>
inline cv::Mat_<cv::Vec<T,4> > create_alpha_mat(const cv::Mat& mat){
    std::vector<cv::Mat> matChannels;
    cv::split(mat, matChannels);

    cv::Mat alpha=cv::Mat(mat.rows,mat.cols, matChannels[0].type());
    alpha.setTo(cv::Scalar(255));
    matChannels.push_back(alpha);

    cv::Mat_<cv::Vec<T,4> > out;
    cv::merge(matChannels, out);
    return out;
}




} //namespace utils
} //namespace radu
