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

#ifndef TINYENGINE_BASE_OPS_H_
#define TINYENGINE_BASE_OPS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <tinyengine/types.h>

#define TN_MAX(A, B) ((A) > (B) ? (A) : (B))
#define TN_MIN(A, B) ((A) < (B) ? (A) : (B))

// bit assignment and check
#define BIT_SET(a, b) ((a) |= (1ULL << (b)))
#define BIT_CLEAR(a, b) ((a) &= ~(1ULL << (b)))
#define BIT_FLIP(a, b) ((a) ^= (1ULL << (b)))
#define BIT_CHECK(a, b) (!!((a) & (1ULL << (b)))) // '!!' to make sure this returns 0 or 1

#define BITMASK_SET(x, mask) ((x) |= (mask))
#define BITMASK_CLEAR(x, mask) ((x) &= (~(mask)))
#define BITMASK_FLIP(x, mask) ((x) ^= (mask))
#define BITMASK_CHECK_ALL(x, mask) (!(~(x) & (mask)))
#define BITMASK_CHECK_ANY(x, mask) ((x) & (mask))

tinyengine_status fully_connected_fp(const float *input,
                                     const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
                                     const uint16_t output_ch,
                                     const float *bias, const float *weights,
                                     float *output);

tinyengine_status stable_softmax_inplace(float *input, const uint16_t length);

tinyengine_status mat_mul_fp(const float *matA, const uint16_t matA_row, const uint16_t matA_col,
                             const float *matB, const uint16_t matB_col, float *output);

#ifdef __cplusplus
}
#endif

#endif /* TINYENGINE_BASE_OPS_H_ */
