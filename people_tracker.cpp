
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;
 
int main (int argc, const char * argv[])
{
	VideoCapture cap("C:\\tltl\\data\\panoramic\\2.avi");
    //VideoCapture cap("C:\\tltl\\data\\panoramic\\3.mp4");
	//VideoCapture cap(CV_CAP_ANY);
	int frameindex = 0;
	int cx = 0, cy = 0;
	string filename = "result.txt";
	ofstream output;
	output.open(filename);

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);    
    if (!cap.isOpened())
        return -1;
 
    Mat img;
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
 
    namedWindow("video capture", CV_WINDOW_AUTOSIZE);
    while (true)
    {
		frameindex ++;
		output<<"Frame"<<" "<<frameindex<<endl;
        cap >> img;
        if (!img.data)
            continue;
 
        vector<Rect> found, found_filtered;
        hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
 
        size_t i, j;
        for (i=0; i<found.size(); i++)
        {
            Rect r = found[i];
            for (j=0; j<found.size(); j++)
                if (j!=i && (r & found[j])==r)
                    break;
            if (j==found.size())
                found_filtered.push_back(r);
        }

        for (i=0; i<found_filtered.size(); i++)
        {
			Rect r = found_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.06);
			r.height = cvRound(r.height*0.9);
			rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
			cx = (r.tl().x+r.br().x)/2;
			cy = (r.tl().y+r.br().y)/2;
			output<<i<<" "<<cx<<" "<<cy<<" "<<r.tl()<<" "<<r.br()<<endl;
		}
        imshow("video capture", img);
        if (waitKey(10) >= 0)
            break;
    }
	output.close();
    return 0;
}
