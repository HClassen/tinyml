/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   convolve.h
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

#ifndef TINYENGINE_BASE_FLOAT_CONVOLVE_H_
#define TINYENGINE_BASE_FLOAT_CONVOLVE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include <tinyengine/types.h>

tinyengine_status_fp group_conv_fp_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row16col_inplace(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																								 const float* filter_data, const float* bias_data, int8_t* output_weight_data, const uint16_t output_height,
																								 const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																								 const float output_activation_max, float* im2col_data, const uint16_t batches, const uint16_t groups,
																								 const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row8col_inplace(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																								const float* filter_data, const float* bias_data, int8_t* output_weight_data, const uint16_t output_height,
																								const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																								const float output_activation_max, float* im2col_data, const uint16_t batches, const uint16_t groups,
																								const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row16col_inplace(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																								 const float* filter_data, const float* bias_data, int8_t* output_weight_data, const uint16_t output_height,
																								 const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																								 const float output_activation_max, float* im2col_data, const uint16_t batches, const uint16_t groups,
																								 const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row8col_inplace(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																								const float* filter_data, const float* bias_data, int8_t* output_weight_data, const uint16_t output_height,
																								const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																								const float output_activation_max, float* im2col_data, const uint16_t batches, const uint16_t groups,
																								const float* scales, const float learning_rate);

#ifdef __cplusplus
}
#endif

#endif /* TINYENGINE_BASE_FLOAT_CONVOLVE_H_ */
