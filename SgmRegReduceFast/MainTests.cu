#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "GenTypes.cu.h"
#include "ParBBHost.cu.h"


//////////////////////////////////
//// MOVE THIS CODE ELSEWHERE ////

//////////////////////////////////


float* initRandArray( unsigned int num_elems ) {
    float* h_in = (float*) malloc(num_elems*sizeof(float));
    std::srand(33); 
    for(unsigned int i=0; i<num_elems; i++) {
        int r = std::rand();
        h_in[i] = ((float)r)/RAND_MAX; 
    }
    return h_in;
}


void validate( float* h_out_small_sgm_cpu
             , float* h_out_small_sgm_gpu
             , const unsigned int num_sgms
             ) {
    unsigned int i = 0;
    float err = 0.0;
    bool is_valid = true;
    for(i = 0; i< num_sgms; i++) {
        err = abs(h_out_small_sgm_gpu[i] - h_out_small_sgm_cpu[i]);
        err = (err <= EPS) ? err : err / h_out_small_sgm_gpu[i]; 
        if (err > EPS) {
            is_valid = false;
            break;
        }
    }
    if(is_valid) printf("VALID SGM-Reduction, YEEEEIIIII!!!\n\n\n");
    else         printf("INVALID SGM-Red, ind: %d, gpu: %.8f, cpu: %.8f, err: %.8f\n\n\n", 
                        i, h_out_small_sgm_gpu[i], h_out_small_sgm_cpu[i], err);
}    

template<class BINOP>
typename BINOP::ElmType* 
testSeqSegReduce(   typename BINOP::ElmType* h_in, 
                    const unsigned int sgm_size,
                    const unsigned int num_sgms
) {
    typename BINOP::ElmType* h_out = (typename BINOP::ElmType*) malloc( num_sgms * sizeof(BINOP::elSize()) );
    struct timeval t_start, t_end, t_diff;
    unsigned long int elapsed;

    gettimeofday(&t_start, NULL);

    for(int i=0; i<num_sgms; i++) {
        typename BINOP::ElmType* sgm = h_in + (i*sgm_size);
        typename BINOP::ElmType acc = BINOP::identity();
        for(int j=0; j<sgm_size; j++) {
            acc = BINOP::apply(acc, sgm[j]);
        }
        h_out[i] = acc;
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("Sequential Segmented-Regular Reduce runs in: %lu microsecs\n", elapsed);
    return h_out;
}
 
template<class BINOP>
typename BINOP::ElmType* 
testParSegReduce(   typename BINOP::ElmType* h_in, 
                    const unsigned int sgm_size,
                    const unsigned int num_sgms
) {

    typedef typename BINOP::ElmType T;
    T *d_in, *d_out;
    const unsigned int inp_mem_size = sgm_size*num_sgms*sizeof(T);
    const unsigned int out_mem_size =          num_sgms*sizeof(T);
    cudaMalloc((void**)&d_in , inp_mem_size);
    cudaMalloc((void**)&d_out, out_mem_size);
    cudaMemcpy(d_in, h_in, inp_mem_size, cudaMemcpyHostToDevice);

    struct timeval t_start, t_end, t_diff;
    unsigned long int elapsed;
    gettimeofday(&t_start, NULL);

    sgmReduce<BINOP>( num_sgms, sgm_size, d_in, d_out );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("Parallel Segmented-Regular Reduce for array size: %d runs in: %lu microsecs\n", num_sgms*sgm_size, elapsed);

    T* h_out = (T*) malloc( out_mem_size );
    cudaMemcpy(h_out, d_out, out_mem_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    return h_out;
}

int main(int argc, char** argv) {
    //const unsigned int num_hwd_thds = 32*1024;
    const unsigned int num_elems    = 50332001; //30332001; //50332001;
    const unsigned int sgm_size = 1000000;//500; //3900000;
    const unsigned int num_sgms = num_elems / sgm_size;

    float* h_in = initRandArray( num_elems );
    float *h_out_small_sgm_cpu, *h_out_small_sgm_gpu;
    
    { // test sequential segmented-regular reduce with small-segment size
        h_out_small_sgm_cpu = testSeqSegReduce<Add<float> >(h_in, sgm_size, num_sgms);
    }

    { // device initialization
        float* tmp_arr;    
        cudaMalloc((void**)&tmp_arr, num_elems * sizeof(float));
        cudaFree(tmp_arr);
        cudaThreadSynchronize();
    }

    { // test sequential segmented-regular reduce with small-segment size
        h_out_small_sgm_gpu = testParSegReduce<Add<float> >(h_in, sgm_size, num_sgms);
    }

    validate(h_out_small_sgm_cpu, h_out_small_sgm_gpu, num_sgms);
    free(h_out_small_sgm_cpu);
    free(h_out_small_sgm_gpu);
    return 1;
}
