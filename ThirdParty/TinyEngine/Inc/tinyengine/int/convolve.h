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

#ifndef TINYENGINE_INT_CONVOLVE_H_
#define TINYENGINE_INT_CONVOLVE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include <tinyengine/types.h>

tinyengine_status convolve_1x1_s8(const q7_t *input,
								  const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
								  const q7_t *kernel,
								  const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
								  const int32_t out_offset,  const int32_t input_offset,
								  const int32_t out_activation_min, const int32_t out_activation_max,
								  q7_t *output,
								  const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
								  q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_fpreq(const q7_t *input,
                                        const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                        const q7_t *kernel,
                                        const int32_t *bias, const float *scales, const int32_t out_offset, const int32_t input_offset,
										const int32_t out_activation_min, const int32_t out_activation_max,
										q7_t *output,
                                        const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
                                        q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch8(const q7_t *input,
									  const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
									  const q7_t *kernel,
									  const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
									  const int32_t out_offset,  const int32_t input_offset,
									  const int32_t out_activation_min, const int32_t out_activation_max,
									  q7_t *output,
									  const uint16_t output_x,  const uint16_t output_y, const uint16_t output_ch,
									  q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch8_fpreq(const q7_t *input,
                                            const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                            const q7_t *kernel,
                                            const int32_t *bias, const float *scales, const int32_t out_offset, const int32_t input_offset,
											const int32_t out_activation_min, const int32_t out_activation_max,
											q7_t *output,
                                            const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
                                            q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch16(const q7_t *input,
									   const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
									   const q7_t *kernel,
									   const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
									   const int32_t out_offset, const int32_t input_offset,
									   const int32_t out_activation_min, const int32_t out_activation_max,
									   q7_t *output,
									   const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
									   q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch16_fpreq(const q7_t *input,
                                             const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                             const q7_t *kernel,
                                             const int32_t *bias, const float *scales, const int32_t out_offset, const int32_t input_offset,
											 const int32_t out_activation_min, const int32_t out_activation_max,
											 q7_t *output,
                                             const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
                                             q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch24(const q7_t *input,
									   const uint16_t input_x, const uint16_t input_y,  const uint16_t input_ch,
									   const q7_t *kernel,
									   const int32_t *bias,  const int32_t *output_shift, const int32_t *output_mult,
									   const int32_t out_offset, const int32_t input_offset,
									   const int32_t out_activation_min, const int32_t out_activation_max,
									   q7_t *output,
									   const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
									   q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch24_fpreq(const q7_t *input,
                                             const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                             const q7_t *kernel,
                                             const int32_t *bias, const float *scales, const int32_t out_offset, const int32_t input_offset,
											 const int32_t out_activation_min, const int32_t out_activation_max,
											 q7_t *output,
                                             const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
                                             q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch48(const q7_t *input,
									   const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
									   const q7_t *kernel,
									   const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
									   const int32_t out_offset, const int32_t input_offset,
									   const int32_t out_activation_min, const int32_t out_activation_max,
									   q7_t *output,
									   const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
									   q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch48_fpreq(const q7_t *input,
                                             const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                             const q7_t *kernel,
                                             const int32_t *bias, const float *scales, const int32_t out_offset, const int32_t input_offset,
											 const int32_t out_activation_min, const int32_t out_activation_max,
											 q7_t *output,
                                             const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
                                             q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_kbuf(const q7_t *input,
                                       const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                       const q7_t *kernel,
                                       const q31_t *kbuf,
									   const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
									   const int32_t out_offset, const int32_t input_offset,
									   const int32_t out_activation_min, const int32_t out_activation_max,
                                       q7_t *output,
									   const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
									   q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_oddch(const q7_t *input,
                                        const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                        const q7_t *kernel,
                                        const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
										const int32_t out_offset, const int32_t input_offset,
										const int32_t out_activation_min, const int32_t out_activation_max,
										q7_t *output,
                                        const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
                                        q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_skip_pad(const q7_t *input,
                                           const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                           const q7_t *kernel,
                                           const int32_t *bias,  const int32_t *output_shift, const int32_t *output_mult,
										   const int32_t out_offset, const int32_t input_offset,
										   const int32_t out_activation_min, const int32_t out_activation_max,
										   q7_t *output,
                                           const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
                                           q15_t *runtime_buf,
                                           const uint16_t pad_t, const uint16_t pad_b, const uint16_t pad_l, const uint16_t pad_r);

tinyengine_status convolve_1x1_s8_SRAM(const q7_t *input,
                                       const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                       const q7_t *kernel,
                                       const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
									   const int32_t out_offset, const int32_t input_offset,
									   const int32_t out_activation_min, const int32_t out_activation_max,
                                       q7_t *output,
									   const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
									   q15_t *runtime_buf, q15_t *kbuf);

tinyengine_status convolve_s8_kernel2x3_inputch3_stride2_pad1(const q7_t *input,
															  const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
															  const q7_t *kernel,
															  const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
															  const int32_t output_offset, const int32_t input_offset,
															  const int32_t output_activation_min, const int32_t output_activation_max,
															  q7_t *output,
															  const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
															  q15_t *runtime_buf, q7_t pad_value);

tinyengine_status convolve_s8_kernel3_inputch3_stride2_pad1(const q7_t *input,
														    const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
															const q7_t *kernel,
															const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
															const int32_t output_offset, const int32_t input_offset,
															const int32_t output_activation_min, const int32_t output_activation_max,
															q7_t *output,
															const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
															q15_t *runtime_buf, q15_t *kbuf, q7_t pad_value);

tinyengine_status convolve_s8_kernel3_inputch3_stride2_pad1_fpreq(const q7_t *input,
                                                                  const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                                                  const q7_t *kernel,
	                                                              const int32_t *bias, const float *scales, const int32_t output_offset, const int32_t input_offset,
	                                                              const int32_t output_activation_min, const int32_t output_activation_max,
                                                                  q7_t *output,
                                                                  const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
                                                                  q15_t *runtime_buf, q15_t *kbuf, q7_t pad_value);

tinyengine_status convolve_s8_kernel3_stride1_pad1(const q7_t *input,
												   const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
												   const q7_t *kernel,
												   const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
												   const int32_t output_offset, const int32_t input_offset,
												   const int32_t output_activation_min, const int32_t output_activation_max,
												   q7_t *output,
												   const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
												   q15_t *runtime_buf, q7_t pad_value);

tinyengine_status convolve_s8_kernel3_stride1_pad1_fpreq(const q7_t *input,
														 const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
														 const q7_t *kernel,
														 const int32_t *bias, const float *scales, const int32_t output_offset, const int32_t input_offset,
														 const int32_t output_activation_min, const int32_t output_activation_max,
														 q7_t *output,
														 const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
														 q15_t *runtime_buf, q7_t pad_value);

/*
 input_x = 90
 input_y = 90
 input_ch = 3
 kernel = weight0
 bias = bias0
 scales = scales0
 output_offset = -128
 input_offset = 128
 output_activation_min = -128
 output_activation_max = 127
 output = buffer0
 output_x = 88
 output_y = 88
 output_ch = 5
 runtime_buf = sbuf
 pad_value = -128
*/

tinyengine_status convolve_s8_kernel3x2_inputch3_stride2_pad1(const q7_t *input,
															  const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
															  const q7_t *kernel,
															  const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
															  const int32_t output_offset, const int32_t input_offset,
															  const int32_t output_activation_min, const int32_t output_activation_max,
															  q7_t *output,
															  const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
															  q15_t *runtime_buf, q15_t *kbuf, q7_t pad_value);

tinyengine_status convolve_u8_kernel3_stride1_pad1(const q8_t *input,
												   const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
												   const q7_t *kernel,
												   const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
												   const int32_t output_offset, const int32_t input_offset,
												   const int32_t output_activation_min, const int32_t output_activation_max,
												   q7_t *output,
												   const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
												   q15_t *runtime_buf, q15_t *kbuf, q7_t pad_value);

tinyengine_status convolve_u8_kernel3_inputch3_stride2_pad1(const q8_t *input,
															const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
															const q7_t *kernel,
															const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
															const int32_t output_offset, const int32_t input_offset,
															const int32_t output_activation_min, const int32_t output_activation_max,
															q7_t *output,
															const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
															q15_t *runtime_buf, q15_t *kbuf, q7_t pad_value);

tinyengine_status convolve_1x1_s8_fpreq_bitmask(const q7_t *input,
                                                const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                                const q7_t *kernel,
                                                const int32_t *bias,
												const float *scales, const int32_t out_offset, const int32_t input_offset,
                                                const int32_t out_activation_min, const int32_t out_activation_max,
                                                q7_t *output,
                                                q7_t *mask,
												const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
                                                q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_fpreq_bitmask_partialCH(const q7_t *input,
                                                          const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                                          const q7_t *kernel_sram, const q7_t *kernel_flash,
                                                          const uint16_t first_k_channel,
                                                          const int32_t *bias, const float *scales, const int32_t out_offset, const int32_t input_offset,
                                                          const int32_t out_activation_min, const int32_t out_activation_max,
                                                          q7_t *output,
                                                          q7_t *mask,
                                                          const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
                                                          q15_t *runtime_buf);

#ifdef __cplusplus
}
#endif

#endif /* TINYENGINE_INT_CONVOLVE_H_ */
