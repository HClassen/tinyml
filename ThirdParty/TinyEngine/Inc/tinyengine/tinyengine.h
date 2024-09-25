/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   tinyengine.h
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

#ifndef TINYENGINE_TINYENGINE_H_
#define TINYENGINE_TINYENGINE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <tinyengine/types.h>
#include <tinyengine/base_ops.h>
#include <tinyengine/gen_nn.h>

#include <tinyengine/int/base_ops.h>
#include <tinyengine/int/convolve.h>
#include <tinyengine/int/patchpadding.h>

#include <tinyengine/float/base_ops.h>
#include <tinyengine/float/convolve.h>
#include <tinyengine/float/pointwise.h>
#include <tinyengine/float/transpose.h>

#ifdef __cplusplus
}
#endif

#endif /* TINYENGINE_TINYENGINE_H_ */
