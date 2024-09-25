/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   types.h
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

#ifndef TINYENGINE_TYPES_H_
#define TINYENGINE_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef int8_t q7_t;
typedef uint8_t q8_t;
typedef int16_t q15_t;
typedef uint16_t q16_t;
typedef int32_t q31_t;
typedef uint32_t q32_t;

typedef enum {
	STATE_SUCCESS = 0,	  /* No error */
	PARAM_NO_SUPPORT = 1, /* Unsupported parameters */
} tinyengine_status;

typedef enum {
	STATE_SUCCESS_fp = 0,	 /* No error */
	PARAM_NO_SUPPORT_fp = 1, /* Unsupported parameters */
} tinyengine_status_fp;

#ifdef __cplusplus
}
#endif

#endif /* TINYENGINE_TYPES_H_ */
