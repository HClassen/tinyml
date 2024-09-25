/* Automatically generated source file */
#include <float.h>
#include <stddef.h>

#include <tinyengine/tinyengine.h>

#include "genModel.h"
#include "genInclude.h"

/* Variables used by all ops */
struct add_params add_params;
int i;
int8_t *int8ptr,*int8ptr2;
int32_t *int32ptr;
float *fptr,*fptr2,*fptr3;

signed char* getInput() {
    return &buffer0[38720];
}
signed char* getOutput() {
    return NNoutput;
}
void end2endinference(q7_t* img){
    invoke(NULL);
}
void invoke(float* labels){
/* layer 0:CONV_2D */
convolve_s8_kernel3_stride1_pad1_fpreq(&buffer0[38720],90,90,3,(const q7_t*) weight0,bias0,scales0,-128,128,-128,127,&buffer0[0],88,88,5,sbuf,-128);
/* layer 1:MAX_POOL_2D */
max_pooling(&buffer0[0],88,88,5,2,2,44,44,-128,127,&buffer0[38720]);
/* layer 2:CONV_2D */
convolve_s8_kernel3_stride1_pad1_fpreq(&buffer0[38720],44,44,5,(const q7_t*) weight1,bias1,scales1,-128,128,-128,127,&buffer0[0],42,42,5,sbuf,-128);
/* layer 3:MAX_POOL_2D */
max_pooling(&buffer0[0],42,42,5,2,2,21,21,-128,127,&buffer0[8820]);
/* layer 4:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[8820],1,1,2205,(const q7_t*) weight2,bias2,scales2,-128,128,-128,127,&buffer0[2208],1,1,64,sbuf);
/* layer 5:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[2208],1,1,64,(const q7_t*) weight3,bias3,scales3,-128,128,-128,127,&buffer0[2272],1,1,64,sbuf);
/* layer 6:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[2272],1,1,64,(const q7_t*) weight4,bias4,scales4,0,128,-128,127,&buffer0[2208],1,1,58,sbuf);
}
void invoke_inf(){
/* layer 0:CONV_2D */
convolve_s8_kernel3_stride1_pad1_fpreq(&buffer0[38720],90,90,3,(const q7_t*) weight0,bias0,scales0,-128,128,-128,127,&buffer0[0],88,88,5,sbuf,-128);
/* layer 1:MAX_POOL_2D */
max_pooling(&buffer0[0],88,88,5,2,2,44,44,-128,127,&buffer0[38720]);
/* layer 2:CONV_2D */
convolve_s8_kernel3_stride1_pad1_fpreq(&buffer0[38720],44,44,5,(const q7_t*) weight1,bias1,scales1,-128,128,-128,127,&buffer0[0],42,42,5,sbuf,-128);
/* layer 3:MAX_POOL_2D */
max_pooling(&buffer0[0],42,42,5,2,2,21,21,-128,127,&buffer0[8820]);
/* layer 4:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[8820],1,1,2205,(const q7_t*) weight2,bias2,scales2,-128,128,-128,127,&buffer0[2208],1,1,64,sbuf);
/* layer 5:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[2208],1,1,64,(const q7_t*) weight3,bias3,scales3,-128,128,-128,127,&buffer0[2272],1,1,64,sbuf);
/* layer 6:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[2272],1,1,64,(const q7_t*) weight4,bias4,scales4,0,128,-128,127,&buffer0[2208],1,1,58,sbuf);
}
