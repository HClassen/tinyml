/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   yolo_output.h
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 * Contact authors:
 *  - Wei-Ming Chen, wmchen@mit.edu
 *  - Wei-Chen Wang, wweichen@mit.edu
 *  - Ji Lin, jilin@mit.edu
 *  - Ligeng Zhu, ligeng@mit.edu
 *  - Song Han, songhan@mit.edu
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#ifndef TINYENGINE_YOLOOUTPUT_H_
#define TINYENGINE_YOLOOUTPUT_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct box {
	float x0;
	float y0;
	float x1;
	float y1;
	float score;
} det_box;

det_box** postprocessing(signed char* input_data[3], signed char y_zero[3], float y_scale[3], unsigned char* data_buf,
						 int w, int h, int output_c, int num_classes, const int anchors[3][3][2], int outputs,
						 const float NMS_threshold, const float VALID_THRESHOLD, int* box_ret, det_box** ret_box);

det_box** postprocessing_fp(float* input_data[3], signed char y_zero[3], float y_scale[3], unsigned char* data_buf,
							int w, int h, int output_c, int num_classes, const int anchors[3][3][2], int outputs,
							const float NMS_threshold, const float VALID_THRESHOLD, int* box_ret, det_box** ret_box);

#ifdef __cplusplus
}
#endif

#endif /* TINYENGINE_YOLOOUTPUT_H_ */
