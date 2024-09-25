/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   pointwise.h
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

#ifndef TINYENGINE_BASE_FLOAT_POINTWISE_H_
#define TINYENGINE_BASE_FLOAT_POINTWISE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include <tinyengine/types.h>

tinyengine_status_fp pointwise_conv_fp_1row10col_10inputdepth_IOHW_int8weight(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																			  const int8_t* filter_data, const float* bias_data, float* output_data, const uint16_t output_height,
																			  const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																			  const float output_activation_max, float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row4col_IOHW_int8weight(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																const int8_t* filter_data, const float* bias_data, float* output_data, const uint16_t output_height,
																const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																const float output_activation_max, float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row4col_IOHW_int8weight_partialCH_8innercol(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																					const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data,
																					float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
																					const float output_activation_min, const float output_activation_max, float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row4col_IOHW_int8weight_partialCH_4innercol(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																					const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data,
																					float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
																					const float output_activation_min, const float output_activation_max, float* im2col_data, const uint16_t batches);

tinyengine_status_fp group_pointwise_conv_fp_in1x1_out1x1_1row10col_uniweight_int8input_inplace(const int8_t* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																								const float* filter_data, const float* bias_data, int8_t* output_weight_data, const uint16_t output_height,
																								const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																								const float output_activation_max, float* im2col_data, const uint16_t batches, const uint16_t groups,
																								const float* scales, const float learning_rate);

tinyengine_status_fp group_pointwise_conv_fp_in1x1_out1x1_1row10col_uniweight_inplace(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
																					  const float* filter_data, const float* bias_data, int8_t* output_weight_data, const uint16_t output_height,
																					  const uint16_t output_width, const uint16_t output_depth, const float output_activation_min,
																					  const float output_activation_max, float* im2col_data, const uint16_t batches, const uint16_t groups,
																					  const float* scales, const float learning_rate);

#ifdef __cplusplus
}
#endif

#endif /* TINYENGINE_BASE_FLOAT_POINTWISE_H_ */
