#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    ifstream infile(argv[1]);
    string filename;
    while (infile >> filename) {
        Mat input_image = imread(filename);
        int width = input_image.cols;
        int height = input_image.rows;
        int new_width, new_height;
        if (width < 256 || height < 256) {
            cout << filename << " " << width << " " << height << endl;
        }
        return 0;
    }


}
