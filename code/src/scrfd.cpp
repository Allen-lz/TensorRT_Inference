#include "../includes/scrfd.h"

scrfd::scrfd(const YAML::Node &config) : Faces(config) {
    anchor_sizes = config["anchor_sizes"].as<std::vector<std::vector<int>>>();
    sum_of_feature = std::accumulate(feature_sizes.begin(), feature_sizes.end(), 0) * anchor_num;

    // std::cout << "sum_of_feature: " << sum_of_feature << std::endl;
    GenerateAnchors();
}

std::vector<FacesRes> scrfd::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<FacesRes> vec_result;
    int index = 0;

    cv::Mat result_matrix_2 = cv::Mat(16800, 15, CV_32FC1, output);
    for (const cv::Mat &src_img : vec_Mat)
    {
        FacesRes result;
        float *out = output + index * outSize;
        float ratio = float(src_img.cols) / float(IMAGE_WIDTH) > float(src_img.rows) / float(IMAGE_HEIGHT)  ? float(src_img.cols) / float(IMAGE_WIDTH) : float(src_img.rows) / float(IMAGE_HEIGHT);

        int result_cols = 2 + bbox_head + landmark_head;
        // std::cout << "result_cols: " << result_cols << std::endl;
        cv::Mat result_matrix = cv::Mat(sum_of_feature, result_cols, CV_32FC1, out);

        for (int item = 0; item < result_matrix.rows; ++item) {
            auto *current_row = result_matrix.ptr<float>(item);
            if(current_row[0] > obj_threshold){
                FaceBox headbox{};
                headbox.bbox.classes = 0;
                headbox.bbox.prob = current_row[0];
                auto *anchor = refer_matrix.ptr<float>(item);
                auto *bbox = current_row + 1;
                auto *keyp = current_row + 2 + bbox_head;
                auto *mask = current_row + 2 + bbox_head + landmark_head;

                headbox.bbox.x = (anchor[0] - bbox[0] * anchor[2]) * ratio;
                headbox.bbox.y = (anchor[1] - bbox[1] * anchor[2]) * ratio;
                headbox.bbox.w = ((bbox[0] + bbox[2]) * anchor[2]) * ratio;
                headbox.bbox.h = ((bbox[1] + bbox[3]) * anchor[2]) * ratio;

                for (int i = 0; i < landmark_head / 2; i++) {
                    KeyPoint point{};
                    point.x = int((anchor[0] + keyp[2 * i] * anchor[2] * landmark_std) * ratio);
                    point.y = int((anchor[1] + keyp[2 * i + 1] * anchor[2] * landmark_std) * ratio);
                    headbox.key_points.key_points.push_back(point);
                }


                if (detect_mask and mask[0] > mask_thresh)
                    headbox.has_mask = true;
                result.faces_results.push_back(headbox);
            }
        }
        NmsDetect(result.faces_results);
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}

void scrfd::GenerateAnchors() {
    refer_matrix = cv::Mat(sum_of_feature, bbox_head, CV_32FC1);
    int line = 0;
    for(size_t feature_map = 0; feature_map < feature_maps.size(); feature_map++) {
        for (int height = 0; height < feature_maps[feature_map][0]; ++height) {
            for (int width = 0; width < feature_maps[feature_map][1]; ++width) {
                for (int anchor = 0; anchor < (int)anchor_sizes[feature_map].size(); ++anchor) {
                    auto *row = refer_matrix.ptr<float>(line);
                    row[0] = (float)width * feature_steps[feature_map];
                    row[1] = (float)height * feature_steps[feature_map];
                    row[2] = feature_steps[feature_map];
                    line++;
                }
            }
        }
    }
}