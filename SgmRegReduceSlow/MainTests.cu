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
    const unsigned int num_elems    = 10332001; //30332001; //50332001;

    //const unsigned int small_sgm_size = 257;
    const unsigned int large_sgm_size = 100000;//500; //3900000;

    float* h_in = initRandArray( num_elems );
    float* h_out_small_sgm_cpu;
    
    { // test sequential segmented-regular reduce with small-segment size
        unsigned int num_sgms = num_elems / large_sgm_size;
        h_out_small_sgm_cpu = testSeqSegReduce<Add<float> >(h_in, large_sgm_size, num_sgms);
        printf("first reduced element: %.8f\n", h_out_small_sgm_cpu[0]);
    }
    
    { // test sequential segmented-regular reduce with small-segment size
        unsigned int num_sgms = num_elems / large_sgm_size;
        h_out_small_sgm_cpu = testParSegReduce<Add<float> >(h_in, large_sgm_size, num_sgms);
        printf("first reduced element: %.8f\n", h_out_small_sgm_cpu[0]);
    }

    { // test sequential segmented-regular reduce with small-segment size
        unsigned int num_sgms = num_elems / large_sgm_size;
        h_out_small_sgm_cpu = testParSegReduce<Add<float> >(h_in, large_sgm_size, num_sgms);
        printf("first reduced element: %.8f\n", h_out_small_sgm_cpu[0]);
    }
    

#if 0
    {
        RedSummary sum1 = findRedKind(num_elems / small_sgm_size, small_sgm_size);
        printf("Block size: %d, Num Segms Per Block: %d\n", sum1.block_size, sum1.num_sgms_blks);

        RedSummary sum2 = findRedKind(num_elems / large_sgm_size, large_sgm_size);
        printf("Block size: %d, Num Blocks Per Segm: %d\n", sum2.block_size, sum2.num_sgms_blks);
    }
#endif

    return 1;
}
