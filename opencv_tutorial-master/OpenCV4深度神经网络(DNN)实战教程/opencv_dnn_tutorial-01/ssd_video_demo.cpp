#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

String objNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };

int main(int argc, char** argv) {
	string bin_model = "D:/projects/opencv_tutorial/data/models/ssd/MobileNetSSD_deploy.caffemodel";
	string protxt = "D:/projects/opencv_tutorial/data/models/ssd/MobileNetSSD_deploy.prototxt";

	// load network model
	Net net = readNetFromCaffe(protxt, bin_model);

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
		Mat blob = blobFromImage(frame, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false, false);
		net.setInput(blob, "data");

		// 执行推理
		Mat detection = net.forward("detection_out");
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
		float confidence_threshold = 0.5;

		// 解析输出数据
		for (int i = 0; i < detectionMat.rows; i++) {
			float* curr_row = detectionMat.ptr<float>(i);
			int image_id = (int)(*curr_row++);
			size_t objIndex = (size_t)(*curr_row++);
			float score = *curr_row++;
			if (score > confidence_threshold) {
				float tl_x = (*curr_row++) * frame.cols;
				float tl_y = (*curr_row++) * frame.rows;
				float br_x = (*curr_row++) * frame.cols;
				float br_y = (*curr_row++) * frame.rows;
				Rect box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
				rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
				putText(frame, format("score: %.2f, %s", score, objNames[objIndex].c_str()), box.tl(), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 0), 2, 8);
			}
		}
		// measure time consume
		vector<double> layersTimings;
		double freq = getTickFrequency() / 1000.0;
		double time = net.getPerfProfile(layersTimings) / freq;
		ostringstream ss;
		ss << "FPS: " << 1000 / time << " ; time : " << time << " ms";

		// show
		putText(frame, ss.str(), Point(20, 20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0), 2, 8);
		imshow("ssd-video-demo", frame);
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