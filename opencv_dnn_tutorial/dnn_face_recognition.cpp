#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;
void recognize_face(Mat& face, Net net, vector<float> &fv);
float compare(vector<float> &fv1, vector<float> &fv2);
int main(int argc, char** argv) {
	string facenet_model = "D:/projects/opencv_tutorial/data/models/face_detector/openface.nn4.small2.v1.t7";
	string pb_model = "D:/opencv_4.1.0/opencv/sources/samples/dnn/face_detector/opencv_face_detector_uint8.pb";
	string pb_txt = "D:/opencv_4.1.0/opencv/sources/samples/dnn/face_detector/opencv_face_detector.pbtxt";

	// load network model
	// Net net = readNetFromCaffe(protxt, bin_model);
	Net net = readNetFromTensorflow(pb_model, pb_txt);
	Net face_net = readNetFromTorch(facenet_model);

	// 设置计算后台
	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(DNN_TARGET_CPU);

	face_net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	face_net.setPreferableTarget(DNN_TARGET_CPU);

	// load face data
	vector<vector<float>> face_data;
	vector<string> labels;
	vector<string> faces;
	glob("D:/my_faces/zhigang", faces);
	for (auto fn : faces) {
		vector<float> fv;
		Mat sample = imread(fn);
		recognize_face(sample, face_net, fv);
		face_data.push_back(fv);
		labels.push_back("zhigang");
	}
	faces.clear();
	glob("D:/my_faces/balvin", faces);
	for (auto fn : faces) {
		vector<float> fv;
		Mat sample = imread(fn);
		recognize_face(sample, face_net, fv);
		face_data.push_back(fv);
		labels.push_back("balvin");
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

				// 获取人脸ROI
				Rect roi;
				roi.x = max(0, box.x);
				roi.y = max(0, box.y);
				roi.width = min(box.width, frame.cols - 1);
				roi.height = min(box.height, frame.rows - 1);
				Mat face = frame(roi);

				// 人脸比对与识别
				vector<float> curr_fv;
				recognize_face(face, face_net, curr_fv);
				// calculate similary
				float minDist = 10;
				int index = -1;
				for (int i = 0; i < face_data.size(); i++) {
					float dist = compare(face_data[i], curr_fv);
					if (minDist > dist) {
						minDist = dist;
						index = i;
					}
				}

				// 显示比较结果
				printf("index : %d, min distance : %.2f \n", index, minDist);
				if (minDist < 0.30 && index >= 0) {
					putText(frame, format("%s", labels[index].c_str()), Point(roi.x, roi.y-10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
				}
				rectangle(frame, box, Scalar(255, 0, 255), 1, 8, 0);
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

void recognize_face(Mat& face, Net net, vector<float> &fv) {
	Mat blob = blobFromImage(face, 1 / 255.0, Size(96, 96), Scalar(0, 0, 0), true, false);
	net.setInput(blob);
	Mat probMat = net.forward();
	Mat vec = probMat.reshape(1, 1);
	// printf("vec rows : %d, vec cols: %d \n", vec.rows, vec.cols);
	for (int i = 0; i < vec.cols; i++) {
		fv.push_back(vec.at<float>(0, i));
	}
}

float compare(vector<float> &fv1, vector<float> &fv2) {
	float dot = 0;
	float sum2 = 0;
	float sum3 = 0;
	for (int i = 0; i < fv1.size(); i++) {
		dot += fv1[i] * fv2[i];
		sum2 += pow(fv1[i], 2);
		sum3 += pow(fv2[i], 2);
	}
	float norm = sqrt(sum2)*sqrt(sum3);
	float similary = dot / norm;
	float dist = acos(similary) / CV_PI;
	return dist;
}