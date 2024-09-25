/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   transpose.h
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

#ifndef TINYENGINE_BASE_FLOAT_TRANSPOSE_H_
#define TINYENGINE_BASE_FLOAT_TRANSPOSE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include <tinyengine/types.h>

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride1_inpad1_outpad0_IOHW_int8weight_partialCH(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																										  const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data,
																										  float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
																										  const float output_activation_min, const float output_activation_max, float* im2col_data, const uint16_t batches,
																										  const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride1_inpad1_outpad0_IOHW_int8weight(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																								const int8_t* filter_data, const float* bias_data, float* output_data, const uint16_t output_height,
																								const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																								const float output_activation_max, float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride2_inpad1_outpad1_IOHW_int8weight_partialCH(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																										  const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data,
																										  float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
																										  const float output_activation_min, const float output_activation_max, float* im2col_data, const uint16_t batches,
																										  const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride2_inpad1_outpad1_IOHW_int8weight(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																								const int8_t* filter_data, const float* bias_data, float* output_data, const uint16_t output_height,
																								const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																								const float output_activation_max, float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride1_inpad2_outpad0_IOHW_int8weight_partialCH(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																										  const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data,
																										  float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
																										  const float output_activation_min, const float output_activation_max, float* im2col_data, const uint16_t batches,
																										  const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride1_inpad2_outpad0_IOHW_int8weight(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																								const int8_t* filter_data, const float* bias_data, float* output_data, const uint16_t output_height,
																								const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																								const float output_activation_max, float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad1_IOHW_int8weight_partialCH(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																										  const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data,
																										  float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
																										  const float output_activation_min, const float output_activation_max, float* im2col_data, const uint16_t batches,
																										  const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad1_IOHW_int8weight(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																								const int8_t* filter_data, const float* bias_data, float* output_data, const uint16_t output_height,
																								const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																								const float output_activation_max, float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride1_inpad3_outpad0_IOHW_int8weight_partialCH(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																										  const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data,
																										  float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
																										  const float output_activation_min, const float output_activation_max, float* im2col_data, const uint16_t batches,
																										  const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride1_inpad3_outpad0_IOHW_int8weight(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																								const int8_t* filter_data, const float* bias_data, float* output_data, const uint16_t output_height,
																								const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																								const float output_activation_max, float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad1_IOHW_int8weight_partialCH(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																										  const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data,
																										  float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
																										  const float output_activation_min, const float output_activation_max, float* im2col_data, const uint16_t batches,
																										  const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad1_IOHW_int8weight(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																								const int8_t* filter_data, const float* bias_data, float* output_data, const uint16_t output_height,
																								const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																								const float output_activation_max, float* im2col_data, const uint16_t batches, const int pad_value);

#ifdef __cplusplus
}
#endif

#endif /* TINYENGINE_BASE_FLOAT_TRANSPOSE_H_ */
