#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int argc, char** argv) {
	string basedir = "D:/projects/opencv_tutorial/data/models/fast_style/";
	string style = "la_muse.t7";

	// load network model
	Net net = readNetFromTorch(format("%s%s", basedir.c_str(), style.c_str()));

	// 设置计算后台
	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// 获取各层信息
	vector<string> layer_names = net.getLayerNames();
	for (int i = 0; i < layer_names.size(); i++) {
		int id = net.getLayerId(layer_names[i]);
		auto layer = net.getLayer(id);
		printf("layer id : %d, type : %s, name : %s \n", id, layer->type.c_str(), layer->name.c_str());
	}
	VideoCapture capture(0);
	Mat frame;
	while (true) {
		bool ret = capture.read(frame);
		if (!ret) break;
		flip(frame, frame, 1);
		// 构建输入
		Mat blob = blobFromImage(frame, 1.0, Size(256, 256), Scalar(103.939, 116.779, 123.68), false, false);
		net.setInput(blob);

		// 执行推理
		Mat out = net.forward();
		
		// 解析输出
		int ch = out.size[1];
		int h = out.size[2];
		int w = out.size[3];
		Mat result = Mat::zeros(Size(w, h), CV_32FC3);
		float* data = out.ptr<float>();

		for (int c = 0; c < ch; c++) {
			for (int row = 0; row < h; row++) {
				for (int col = 0; col < w; col++) {
					result.at<Vec3f>(row, col)[c] = *data++;
				}
			}
		}
		add(result, Scalar(103.939, 116.779, 123.68), result);
		result /= 255.0;

		// 中值滤波
		medianBlur(result, result, 5);
		Mat dst;
		resize(result, dst, frame.size());
		// measure time consume
		vector<double> layersTimings;
		double freq = getTickFrequency() / 1000.0;
		double time = net.getPerfProfile(layersTimings) / freq;
		ostringstream ss;
		ss << "FPS: " << 1000 / time << " ; time : " << time << " ms";

		// show
		putText(dst, ss.str(), Point(20, 20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0), 2, 8);
		imshow("style-transfer-demo", dst);
		char c = waitKey(1);
		if (c == 27) { // ESC
			break;
		}
	}

	// 释放资源
	capture.release();
	waitKey(0);
	return 0;
}