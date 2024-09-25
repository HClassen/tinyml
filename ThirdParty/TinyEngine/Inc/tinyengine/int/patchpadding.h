/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   patchpadding.h
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

#ifndef TINYENGINE_INT_PATCHPADDING_H_
#define TINYENGINE_INT_PATCHPADDING_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include <tinyengine/types.h>

tinyengine_status patchpadding_convolve_s8_kernel3_inputch3_stride2(const q7_t *input,
																	const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
																	const q7_t *kernel,
																	const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
																	const int32_t output_offset, const int32_t input_offset,
																	const int32_t output_activation_min, const int32_t output_activation_max,
																	q7_t *output,
																	const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
																	q15_t *runtime_buf, q15_t *kbuf,
																	q7_t pad_value, const uint16_t pad_t, const uint16_t pad_b, const uint16_t pad_l, const uint16_t pad_r);

tinyengine_status patchpadding_depthwise_kernel3x3_stride1_inplace_CHW(q7_t *input,
																	   const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
																	   const q7_t *kernel,
																	   const int32_t *bias, const int32_t *biasR, const int32_t *output_shift, const int32_t *output_mult,
																	   const int32_t output_offset, const int32_t input_offset,
																	   const int32_t output_activation_min, const int32_t output_activation_max,
																	   q7_t *output,
																	   const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
																	   q15_t *runtime_buf,
																	   q7_t pad_value, const uint16_t pad_t, const uint16_t pad_b, const uint16_t pad_l, const uint16_t pad_r);

tinyengine_status patchpadding_depthwise_kernel3x3_stride2_inplace_CHW(q7_t *input,
																	   const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
																	   const q7_t *kernel,
																	   const int32_t *bias, const int32_t *biasR, const int32_t *output_shift, const int32_t *output_mult,
																	   const int32_t output_offset, const int32_t input_offset,
																	   const int32_t output_activation_min, const int32_t output_activation_max,
																	   q7_t *output,
																	   const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
																	   q15_t *runtime_buf,
																	   q7_t pad_value, const uint16_t pad_t, const uint16_t pad_b, const uint16_t pad_l, const uint16_t pad_r);

tinyengine_status patchpadding_kbuf_convolve_s8_kernel3_inputch3_stride2(const q7_t *input,
																		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
																		const q7_t *kernel,
																		const q31_t *kbuf,
																		const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
																		const int32_t output_offset, const int32_t input_offset,
																		const int32_t output_activation_min, const int32_t output_activation_max,
																		q7_t *output,
																		const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
																		q15_t *runtime_buf,
																		q7_t pad_value, const uint16_t pad_t, const uint16_t pad_b, const uint16_t pad_l, const uint16_t pad_r);

#ifdef __cplusplus
}
#endif

#endif /* TINYENGINE_INT_PATCHPADDING_H_ */
