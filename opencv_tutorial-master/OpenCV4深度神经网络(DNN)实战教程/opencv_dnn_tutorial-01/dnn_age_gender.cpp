#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

String  ageList[] = { "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)" };
String genderList[] = { "Male", "Female" };
int main(int argc, char** argv) {
	string pb_model = "D:/opencv_4.1.0/opencv/sources/samples/dnn/face_detector/opencv_face_detector_uint8.pb";
	string pb_txt = "D:/opencv_4.1.0/opencv/sources/samples/dnn/face_detector/opencv_face_detector.pbtxt";
	
	string age_model = "D:/projects/opencv_tutorial/data/models/cnn_age_gender_models/age_net.caffemodel";
	string age_protxt = "D:/projects/opencv_tutorial/data/models/cnn_age_gender_models/age_deploy.prototxt";
	string gender_model = "D:/projects/opencv_tutorial/data/models/cnn_age_gender_models/gender_net.caffemodel";
	string gender_protxt = "D:/projects/opencv_tutorial/data/models/cnn_age_gender_models/gender_deploy.prototxt";

	// load face
	Net net = readNetFromTensorflow(pb_model, pb_txt);
	// load age and gender
	Net age_net = readNetFromCaffe(age_protxt, age_model);
	Net gender_net = readNetFromCaffe(gender_protxt, gender_model);

	// 设置计算后台
	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(DNN_TARGET_CPU);
	age_net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	age_net.setPreferableTarget(DNN_TARGET_CPU);
	gender_net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	gender_net.setPreferableTarget(DNN_TARGET_CPU);

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
		int padding = 15;
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
				roi.x = max(0, box.x - padding);
				roi.y = max(0, box.y - padding);
				roi.width = min(box.width + padding, frame.cols - 1);
				roi.height = min(box.height + padding, frame.rows - 1);
				Mat face = frame(roi);

				// 执行推断年龄与性别预测
				Mat face_blob = blobFromImage(face, 1.0, Size(227, 227), Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);
				age_net.setInput(face_blob);
				gender_net.setInput(face_blob);
				Mat ageProbs = age_net.forward();
				Mat genderProbs = gender_net.forward();

				// 解析输出结果
				Mat prob_age = ageProbs.reshape(1, 1);
				Point classNum;
				double classProb;
				minMaxLoc(prob_age, 0, &classProb, 0, &classNum);
				int classIdx = classNum.x;
				String age = ageList[classIdx];

				Mat prob_gender = genderProbs.reshape(1, 1);
				minMaxLoc(prob_gender, 0, &classProb, 0, &classNum);
				classIdx = classNum.x;
				String gender = genderList[classIdx];
				rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
				putText(frame, format("age:%s, gender:%s", age.c_str(), gender.c_str()), box.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1, 8);
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