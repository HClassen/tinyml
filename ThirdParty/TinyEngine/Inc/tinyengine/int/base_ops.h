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

#ifndef TINYENGINE_INT_BASE_OPS_H_
#define TINYENGINE_INT_BASE_OPS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include <tinyengine/types.h>

struct add_params {
	int input_h, input_w, input_c, left_shift;
	int input1_offset, input1_multiplier, input1_shift;
	int input2_offset, input2_multiplier, input2_shift;
	int output_offset, output_multiplier, output_shift;
	int quantized_activation_max, quantized_activation_min;
};

tinyengine_status add(int size, struct add_params *params,
					  const int8_t *input1_data, const int8_t *input2_data,
					  int8_t *output_data);

tinyengine_status add_fpreq(int size,
                            const int8_t *input1_data, const float input1_scale, const float input1_zero,
							const int8_t *input2_data, const float input2_scale, const float input2_zero,
							const float output_scale, const float zero_y, int8_t *output_data);

tinyengine_status add_fpreq_mask(int size,
                                 const int8_t *input1_data, const float input1_scale, const float input1_zero,
								 const int8_t *input2_data, const float input2_scale, const float input2_zero,
								 const float output_scale, const float zero_y, int8_t *output_data,
								 int8_t *output_mask);

tinyengine_status add_fpreq_bitmask(int size,
                                    const int8_t *input1_data, const float input1_scale, const float input1_zero,
                                    const int8_t *input2_data, const float input2_scale, const float input2_zero,
                                    const float output_scale, const float zero_y,
									int8_t *output_data, int8_t *output_mask);

tinyengine_status avg_pooling(const q7_t *input,
							  const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
							  const uint16_t sample_h, const uint16_t sample_w,
							  const uint16_t output_h, const uint16_t output_w,
							  const int32_t out_activation_min, const int32_t out_activation_max,
							  q7_t *output);

tinyengine_status max_pooling(const q7_t* input,
							  const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
							  const uint16_t sample_h, const uint16_t sample_w,
							  const uint16_t output_h, const uint16_t output_w,
							  const int32_t out_activation_min, const int32_t out_activation_max,
							  q7_t* output);

tinyengine_status element_mult_nx1(const q7_t *input,
                                   const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
                                   const q7_t *input2,
                                   const int16_t input1_offset,  const int16_t input2_offset,
                                   const int16_t output_offset,
								   const int32_t out_activation_min, const int32_t out_activation_max, const float output_scale,
                                   q7_t *output);

tinyengine_status concat_ch(const q7_t *input1,
                            const uint16_t input_x, const uint16_t input_y, const uint16_t input1_ch,
                            const q7_t *input2,
                            const uint16_t input2_ch,
							q7_t *output);

#ifdef __cplusplus
}
#endif

#endif /* TINYENGINE_INT_BASE_OPS_H_ */
