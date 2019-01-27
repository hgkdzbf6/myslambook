#include <iostream> 
#include <ctime>
#include <iomanip>
#include <chrono>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define MATRIX_SIZE 50

int main(int argc, char const *argv[])
{
    cv::Mat image = cv::imread("../pics/left0.jpg");
    if ( image.data == nullptr){
        cerr << "文件不存在" << endl;
        return -1;
    }
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    for(size_t y=0;y<image.rows;y++){
        for(size_t x=0;x<image.cols;x++){
            unsigned char* row_ptr = image.ptr<unsigned char>(y);
            unsigned char* data_ptr = &row_ptr[x*image.channels()];
            for(int c=0;c!=image.channels();c++){
                unsigned char data = data_ptr[c];
            }
        }
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << time_used.count() << endl;
    return 0;
}
