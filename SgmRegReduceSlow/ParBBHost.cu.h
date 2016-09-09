#ifndef PAR_BB_HOST
#define PAR_BB_HOST

#include "ParBBKernels.cu.h"

#include <sys/time.h>
#include <time.h> 

#define MAX_BLOCK_SIZE  512 //1024
#define MIN_BLOCK_SIZE  128 //128
#define WARP_SIZE       32

int nextMultOf(unsigned int x, unsigned int m) {
    if( x % m ) return x - (x % m) + m;
    else        return x;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

enum SgmRedType { SmallSgm, LargeSgm };
struct RedSummary {
    SgmRedType      red_type;
    unsigned int    block_size;
    unsigned int    num_sgms_blks;
};

RedSummary findRedKind(const unsigned int num_sgms, const unsigned int sgm_size) {
    const unsigned int num_elems = num_sgms * sgm_size;
    RedSummary res;
    if (sgm_size < MAX_BLOCK_SIZE) {
        unsigned int mul_warp = MIN_BLOCK_SIZE / WARP_SIZE;
        // find the first legal block-size greater than 
        while( (mul_warp*WARP_SIZE) < sgm_size) { mul_warp++; }
        
        unsigned int waste = num_elems, lowest_waste = waste, best_mul_warp = mul_warp;
        while( (mul_warp*WARP_SIZE) <= MAX_BLOCK_SIZE ) {
            unsigned int block_size    = mul_warp*WARP_SIZE;
            unsigned int elems_per_block = (block_size / sgm_size)*sgm_size;
            unsigned int tot_threads = ceil( ((float)num_elems) / elems_per_block ) * block_size;
            waste = tot_threads - num_elems;
            if(waste < lowest_waste) {
                lowest_waste = waste; best_mul_warp = mul_warp; 
            }
            mul_warp ++;
        }
        res.red_type      = SmallSgm;
        res.block_size    = best_mul_warp  * WARP_SIZE;
        res.num_sgms_blks = res.block_size / sgm_size;
        return res;
    } else {
        unsigned int mul_warp = MAX_BLOCK_SIZE / WARP_SIZE;
        
        unsigned int waste = num_elems, lowest_waste = waste, best_mul_warp = mul_warp;
        while( mul_warp*WARP_SIZE >= MIN_BLOCK_SIZE ) {
            unsigned int block_size = mul_warp * WARP_SIZE;
            unsigned int blocks_per_sgm = ceil( ((float)sgm_size) / block_size );
            unsigned int tot_threads    = blocks_per_sgm * num_sgms * block_size;
            waste = tot_threads - num_elems;
            if(waste < lowest_waste) {
                lowest_waste = waste; best_mul_warp = mul_warp; 
            }
            mul_warp --;
        }
        res.red_type      = LargeSgm;
        res.block_size    = best_mul_warp  * WARP_SIZE;
        res.num_sgms_blks = ceil( ((float)sgm_size) / res.block_size );
        return res;
    }
} 


template<class OP>
void sgmReduce( unsigned int num_sgms
              , unsigned int sgm_size
              , typename OP::ElmType* d_in   // device [num_sgms,sgm_size]
              , typename OP::ElmType* d_out  // device [num_sgms]
) {
    typedef typename OP::ElmType T;
    T* inp_arr = d_in;
    T* tmp_arr = NULL;
    T* out_arr;
    RedSummary summary = findRedKind(num_sgms, sgm_size);

    unsigned int tmp_size;
    { // temporary space is bounded by ( N/MB * ( MB / (MB-1) ) ),
      // where MB is MIN_BLOCK_SIZE and N is total number of elements.
        float tmp = ((float)MIN_BLOCK_SIZE) / (MIN_BLOCK_SIZE-1);
        tmp = tmp / MIN_BLOCK_SIZE;
        tmp = tmp * (num_sgms * sgm_size);
        tmp_size = ceil(tmp) * 2;
    }
#if 1
    cudaMalloc((void**)&tmp_arr, tmp_size * sizeof(T));

    unsigned int offset = 0;
    while(summary.red_type != SmallSgm) {
//        printf("LARGE SGM, block_size: %d, blocks_per_sgm: %d\n", summary.block_size, summary.num_sgms_blks);
        // block_size num_sgms_blks ceil( ((float)num_sgms) / summary.num_sgms_blks )
        unsigned int sh_mem_size = summary.block_size * sizeof(T);
        unsigned int num_blocks  = num_sgms * summary.num_sgms_blks;
        out_arr = tmp_arr + offset;
        if( (offset+num_blocks) > (tmp_size/2) ) {
            printf("ALLOCATION ERROR: tmp_alloc_size: %d, offset: %d, num_blocks: %d", 
                    tmp_size/2, offset, num_blocks);
            exit(0);
        }
        offset += num_blocks;
        //cudaMalloc((void**)&out_arr, num_blocks*sizeof(typename OP::ElmType));
        
        sgmRedLargeKernel<OP><<< num_blocks, summary.block_size, sh_mem_size >>>
            (inp_arr, out_arr, sgm_size, summary.num_sgms_blks);

        inp_arr  = out_arr;
        sgm_size = summary.num_sgms_blks;
        summary = findRedKind(num_sgms, sgm_size);
    }
#endif
    { // last step: we are in the case (summary.red_type == SmallSgm)!
//        printf("SMALL SGM, block_size: %d, blocks_per_sgm: %d, sgm_size: %d, num_sgms: %d\n", 
//                summary.block_size, summary.num_sgms_blks, sgm_size, num_sgms);

        unsigned int sh_mem_size = summary.block_size * (sizeof(T) + sizeof(int));
        unsigned int num_blocks  = ceil( ((float)num_sgms) / summary.num_sgms_blks);

//    struct timeval t_start, t_end, t_diff;
//    unsigned long int elapsed;
//    gettimeofday(&t_start, NULL);

        sgmRedSmallKernel<OP><<< num_blocks, summary.block_size, sh_mem_size >>>
            (inp_arr, d_out, num_sgms, sgm_size, summary.num_sgms_blks);
        cudaThreadSynchronize();

//    gettimeofday(&t_end, NULL);
//    timeval_subtract(&t_diff, &t_end, &t_start);
//    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
//    printf("Parallel111 Segmented-Regular Reduce for array size: %d runs in: %lu microsecs\n", num_sgms*sgm_size, elapsed);


//        cudaError_t err = cudaGetLastError();
//        if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
                
    }

    cudaFree(tmp_arr);
}
#endif //PAR_BB_HOST
