/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   base_ops.h
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

#ifndef TINYENGINE_BASE_FLOAT_OPS_H_
#define TINYENGINE_BASE_FLOAT_OPS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#include <tinyengine/types.h>

tinyengine_status_fp add_fp(const uint16_t size, const float* input1_data, const float* input2_data,
							float* output_data);

tinyengine_status_fp div_fp(const uint16_t size, const float* input1_data, const float* input2_data,
							float* output_data);

tinyengine_status_fp less(const uint16_t size, const float* input1_data, const float* input2_data, bool* output_data);

tinyengine_status_fp LogSoftmax(const float* input_data, const uint16_t input_height, const uint16_t input_width,
								const uint16_t input_depth, float* output_data, const uint16_t output_height,
								const uint16_t output_width, const uint16_t output_depth);

tinyengine_status_fp mul(const uint16_t size, const float* input1_data, const float* input2_data, float* output_data);

tinyengine_status_fp negative(const uint16_t size, const float* input1_data, bool* output_data);

tinyengine_status_fp nll_loss(const float* input_data, const uint16_t input_dim, const uint16_t input_depth,
							  const float* target, const uint16_t target_size, float* output_data);

tinyengine_status_fp strided_slice_3Dto3D(const float* input, const uint16_t input_h, const uint16_t input_w,
										  const uint16_t input_c, const uint16_t* begin, const uint16_t* end,
										  const uint16_t* stride, float* output, const uint16_t output_h,
										  const uint16_t output_w, const uint16_t output_c);

tinyengine_status_fp strided_slice_4Dto4D(const float* input, const uint16_t inn, const uint16_t inc,
										  const uint16_t inh, const uint16_t inw, const uint16_t* begin,
										  const uint16_t* end, const uint16_t* stride, float* output, const uint16_t on,
										  const uint16_t oc, const uint16_t oh, const uint16_t ow);

tinyengine_status_fp sub(const uint16_t size, const float* input1_data, const float* input2_data, float* output_data);

tinyengine_status_fp sum_2D(const float* input_data, const uint16_t matA_row, const uint16_t matA_col,
							const uint16_t axis, float* output_data);

tinyengine_status_fp sum_3D(const float* input_data, const uint16_t input_w, const uint16_t input_h,
							const uint16_t input_c, const uint16_t axis, float* output_data);

tinyengine_status_fp sum_4D_exclude(const float* input_data, const uint16_t d1, const uint16_t d2, const uint16_t d3,
									const uint16_t d4, const uint16_t axis, float* output_data);

tinyengine_status_fp tte_exp(const uint16_t size, const float* input_data, float* output_data);

tinyengine_status_fp where(const bool* inMask, const uint16_t size, const float* input1_data, const float* input2_data,
						   float* output_data);

tinyengine_status_fp where_zeros(const bool* inMask, const uint16_t size, const float* input1_data, float* output_data);

tinyengine_status_fp where_zeros_inplace(const bool* inMask, const uint16_t size, float* input1_data);

tinyengine_status_fp where_zeros_inplace_bit(const unsigned char* inMask, const uint16_t size, float* input1_data);

#ifdef __cplusplus
}
#endif

#endif /* TINYENGINE_BASE_FLOAT_OPS_H_ */
