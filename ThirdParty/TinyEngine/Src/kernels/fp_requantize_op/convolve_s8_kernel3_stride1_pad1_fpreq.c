/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   convolve_s8_kernel3_inputch3_stride2_pad1_fpreq.c
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

#include <stdio.h>

#include <tinyengine/types.h>
#include <tinyengine/base_ops.h>

#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "img2col_element.h"

#define INDEX(x, y, z, n_rows, n_channels)	\
	(x) * (n_rows) * (n_channels) + (y) * (n_channels) + (z)

static q7_t convolve(const q15_t input[3][3], const q7_t *kernel,
					 const int32_t bias, const float scale, const int32_t output_offset,
					 const int32_t output_activation_min, const int32_t output_activation_max) {
	q31_t sum = bias;

	for (int x = 0; x < 3; x++) {
		for (int y = 0; y < 3; y++) {
			sum += input[x][y] * kernel[x * 3 + y];
		}
	}

	sum = (q31_t)((float)sum * scale);
	sum += output_offset;
	sum = MAX(sum, output_activation_min);
	sum = MIN(sum, output_activation_max);

	return (q7_t)sum;
}

tinyengine_status convolve_s8_kernel3_stride1_pad1_fpreq(const q7_t *input, const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch, const q7_t *kernel,
														 const int32_t *bias, const float *scales, const int32_t output_offset, const int32_t input_offset,
														 const int32_t output_activation_min, const int32_t output_activation_max, q7_t *output, const uint16_t output_x,
														 const uint16_t output_y, const uint16_t output_ch, q15_t *runtime_buf, q7_t pad_value) {
	q15_t matrix[3][3];

	for (int y = 0; y < output_y; y++) {
		for (int x = 0; x < output_x; x++) {
			for (int c = 0; c < input_ch; c++) {
				matrix[0][0] = input[INDEX(x - 1, y - 1, c, input_x, input_ch)] + input_offset;
				matrix[0][1] = input[INDEX(x - 1, y, c, input_x, input_ch)] + input_offset;
				matrix[0][2] = input[INDEX(x - 1, y + 1, c, input_x, input_ch)] + input_offset;

				matrix[1][0] = input[INDEX(x, y - 1, c, input_x, input_ch)] + input_offset;
				matrix[1][0] = input[INDEX(x, y, 0, input_x, input_ch)] + input_offset;
				matrix[1][0] = input[INDEX(x, y + 1, c, input_x, input_ch)] + input_offset;

				matrix[2][0] = input[INDEX(x + 1, y - 1, c, input_x, input_ch)] + input_offset;
				matrix[2][0] = input[INDEX(x + 1, y, c, input_x, input_ch)] + input_offset;
				matrix[2][0] = input[INDEX(x + 1, y + 1, c, input_x, input_ch)] + input_offset;

				for (int i = 0; i < output_ch; i++) {
					q7_t sum = convolve(matrix, &kernel[i * output_ch], bias[i], scales[i],
										output_offset, output_activation_min, output_activation_max);

					if (c == 0)
						output[INDEX(y, x, i, output_x, output_ch)] = sum;
					else
						output[INDEX(y, x, i, output_x, output_ch)] += sum;
				}
			}
		}
	}

	/* Return to application */
	return STATE_SUCCESS;
}
