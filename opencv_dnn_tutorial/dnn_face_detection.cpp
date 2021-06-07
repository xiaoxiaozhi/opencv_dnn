#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int argc, char** argv) {
	// string bin_model = "D:/opencv_4.1.0/opencv/sources/samples/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel";
	// string protxt = "D:/opencv_4.1.0/opencv/sources/samples/dnn/face_detector/deploy.prototxt";

	string pb_model = "D:/opencv_4.1.0/opencv/sources/samples/dnn/face_detector/opencv_face_detector_uint8.pb";
	string pb_txt = "D:/opencv_4.1.0/opencv/sources/samples/dnn/face_detector/opencv_face_detector.pbtxt";

	// load network model
	Net net = readNetFromTensorflow(pb_model, pb_txt);

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
		Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
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
				putText(frame, format("score: %.2f", score), box.tl(), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 0), 2, 8);
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
		imshow("face-detection-demo", frame);
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