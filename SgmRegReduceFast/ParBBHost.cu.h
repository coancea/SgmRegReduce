#ifndef PAR_BB_HOST
#define PAR_BB_HOST

#include <assert.h>

#include "ParBBKernels.cu.h"

#include <sys/time.h>
#include <time.h> 

#define OPT_WASTE_PER   1.5
#define OPT_HWD_THDS    65536
#define WARP_SIZE       32
#define MAX_BLOCK_SIZE  768 //1024 //512 //1024
#define MIN_BLOCK_SIZE  96 //128
#define MIN_MUL_WARP    (MIN_BLOCK_SIZE/WARP_SIZE)
#define MAX_MUL_WARP    (MAX_BLOCK_SIZE/WARP_SIZE)
#define TILE_SIZE       32
#define EPS             0.000033

#define DEBUG_TIMING

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

/**
 * d_in       is the device matrix; it is supposably
 *                allocated and holds valid values (input).
 *                semantically of size [height x width]
 * d_out      is the output GPU array -- if you want 
 *            its data on CPU needs to copy it back to host.
 *                semantically of size [width x height]
 * height     is the height of the input matrix
 * width      is the width  of the input matrix
 */ 

template<class T, int tile>
void transposePad(  T*                 inp_d,  
                    T*                 out_d, 
                    const unsigned int height, 
                    const unsigned int width,
                    const unsigned int oinp_size,
                    T                  pad_elem
) {
   // 1. setup block and grid parameters
   int  dimy = ceil( ((float)height)/tile ); 
   int  dimx = ceil( ((float) width)/tile );
   dim3 block(tile, tile, 1);
   dim3 grid (dimx, dimy, 1);
 
   //2. execute the kernel
   matTransposeTiledPadKer<T,tile> <<< grid, block >>>
    (inp_d, out_d, height, width, oinp_size,pad_elem); 
   
    //cudaThreadSynchronize();
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

enum SgmRedType { SmallSgm, LargeSgm, SeqSgm };
struct RedSummary {
    SgmRedType      red_type;
    unsigned int    chunk_size;
    unsigned int    block_size;
    unsigned int    num_sgms_blks;
};

void printRedSummary(RedSummary res) {
    if        (res.red_type == SeqSgm) {
        printf("SEQun-Sgm Summary: chunk_size: %d, block_size: %d, num_sgms_per_block: %d\n", 
                res.chunk_size, res.block_size, res.num_sgms_blks);
    } else if (res.red_type == SmallSgm) {
        printf("Small-Sgm Summary: chunk_size: %d, block_size: %d, num_sgms_per_block: %d\n", 
                res.chunk_size, res.block_size, res.num_sgms_blks);
    } else {
        printf("Large-Sgm Summary: chunk_size: %d, block_size: %d, num_blks_per_segm : %d\n", 
                res.chunk_size, res.block_size, res.num_sgms_blks);
    }
}

void BEST_CHUNK_BLOCK_SZ(   unsigned int  num_elems
                        ,   unsigned int& lowest_waste 
                        ,   unsigned int& best_mul_warp
                        ,   unsigned int& best_chunk 
                        ,   unsigned int  waste
                        ,   unsigned int  mul_warp
                        ,   unsigned int  chunk
) {
    unsigned int red_factor      = chunk*mul_warp;
    unsigned int best_red_factor = best_chunk*best_mul_warp;
    float waste_per = (100.0 * waste)  / num_elems;
    if( (red_factor >= best_red_factor && waste <= lowest_waste) ||   // (1)   
        (red_factor >= best_red_factor && waste_per <= OPT_WASTE_PER) // (2)
      ) {
        // either clear-cut case OR 
        // "negligible" waste => choose highest chunk factor!
        if(red_factor > best_red_factor || chunk > best_chunk) {
            lowest_waste = waste; best_chunk = chunk; best_mul_warp = mul_warp;
        }
    } else {
        // nothing, remains the current best ...
    }
}

RedSummary findMultSegmsPerBlock(   const unsigned int num_sgms 
                                ,   const unsigned int sgm_size 
) {
    RedSummary res;
    const unsigned int num_elems = num_sgms * sgm_size;
    const unsigned int mul_warp_min1 = MIN_BLOCK_SIZE / WARP_SIZE;
    const unsigned int mul_warp_min2 = ceil( ((float)sgm_size) / WARP_SIZE);
    unsigned int mul_warp = max( mul_warp_min1, mul_warp_min2 );
        
    unsigned int waste = num_elems, lowest_waste = waste, best_mul_warp = 0;
    for( ; mul_warp <= MAX_MUL_WARP; mul_warp++ ) {
        unsigned int block_size = mul_warp*WARP_SIZE;
        unsigned int elems_per_block = (block_size / sgm_size) * sgm_size;
        unsigned int tot_threads = ceil( ((float)num_elems) / elems_per_block ) * block_size;
        waste = tot_threads - num_elems;
        if(waste < lowest_waste) {
            lowest_waste = waste; best_mul_warp = mul_warp; 
        }
    }

    res.red_type      = SmallSgm;
    res.chunk_size    = 1;
    res.block_size    = best_mul_warp  * WARP_SIZE;
    res.num_sgms_blks = res.block_size / sgm_size;
    return res;
}


RedSummary findMultBlocksPerSegm(   const unsigned int num_sgms 
                                ,   const unsigned int sgm_size 
) {
    const unsigned int num_elems = num_sgms * sgm_size;
    RedSummary res;

//    assert(sgm_size >= MAX_BLOCK_SIZE && 
//            "INTERNAL ERROR: not multiple-blocks-per-segm case!");

    const unsigned int MAX_CHUNK = min(sgm_size / MIN_BLOCK_SIZE, 
                                       num_sgms*sgm_size / OPT_HWD_THDS );

    unsigned int chunk, best_chunk = 0;
    unsigned int mul_warp, best_mul_warp = 0;
    unsigned int waste = num_elems, lowest_waste = num_elems;

    for(chunk = 1; chunk < MAX_CHUNK; chunk++) {
        for(mul_warp = MIN_BLOCK_SIZE / WARP_SIZE; 
            mul_warp <= MAX_BLOCK_SIZE / WARP_SIZE; mul_warp++  ) {

            unsigned int block_size = mul_warp * WARP_SIZE;
            unsigned int blocks_per_sgm = ceil( ((float)sgm_size) / (block_size*chunk) );
            unsigned int tot_threads    = blocks_per_sgm * num_sgms * block_size * chunk;
            waste = tot_threads - num_elems;

            // reduction part with in-place updates
            BEST_CHUNK_BLOCK_SZ(num_elems,
                                lowest_waste, best_mul_warp, best_chunk, 
                                       waste,      mul_warp,      chunk);
        }
    }

    res.red_type      = LargeSgm;
    res.chunk_size    = best_chunk;
    res.block_size    = best_mul_warp * WARP_SIZE;
    res.num_sgms_blks = (res.block_size == 0 || res.chunk_size == 0) ? 0 :
                        ceil( ((float)sgm_size) / (res.block_size*res.chunk_size) );
    return res;
}


RedSummary findRedKind(const unsigned int num_sgms, const unsigned int sgm_size) {
    RedSummary res;

    if (OPT_HWD_THDS <= num_sgms) {
        res.red_type   = SeqSgm;
        res.chunk_size = sgm_size;
        res.block_size = MIN_BLOCK_SIZE * 2;
        res.num_sgms_blks = res.block_size;
    } else {
        res = findMultBlocksPerSegm(num_sgms, sgm_size);

        if (res.block_size == 0) { // invalid LargeSgm
            if(sgm_size < MAX_BLOCK_SIZE) {
                // OK! Case SmallSgm
                res = findMultSegmsPerBlock(num_sgms, sgm_size);
                if (res.block_size < MIN_BLOCK_SIZE) {
                    printf("ERROR: INVALID SmallSgm block size!!!\n");
                    exit(0);
                }
            } else {
                printf("Invalid Case: sgm_size == %d >= MAX_BLOCK_SIZE == %d, but LargeSgm is INVALID!!!\n",
                        sgm_size, MAX_BLOCK_SIZE);
                exit(0);
            }
        }
    }
#ifdef DEBUG_TIMING
    printRedSummary(res);
#endif
    return res;
}


template<class OP>
void sgmReduce( unsigned int num_sgms
              , unsigned int sgm_size
              , typename OP::ElmType* d_in   // device [num_sgms,sgm_size]
              , typename OP::ElmType* d_out  // device [num_sgms]
) {
    const unsigned int num_elems = num_sgms * sgm_size;
    typedef typename OP::ElmType T;
    T* inp_arr = d_in;
    T* tmp_arr = NULL;
    T* trnsp_arr = NULL;
    T* out_arr;

    unsigned int tmp_size;
    { // temporary space is bounded by ( N/MB * ( MB / (MB-1) ) ),
      // where MB is MIN_BLOCK_SIZE and N is total number of elements.
        float tmp = ((float)MIN_BLOCK_SIZE) / (MIN_BLOCK_SIZE-1);
        tmp = tmp / MIN_BLOCK_SIZE;
        tmp = tmp * (num_sgms * sgm_size);
        tmp_size = ceil(tmp) * 2;
    }

#ifdef DEBUG_TIMING
        struct timeval t_start, t_end, t_diff;
        unsigned long int elapsed;
        gettimeofday(&t_start, NULL);
#endif
        
    RedSummary summary = findRedKind(num_sgms, sgm_size);

#ifdef DEBUG_TIMING

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("findRedKernel runs in: %lu microsecs\n", elapsed);
#endif


    if(summary.red_type == LargeSgm && summary.num_sgms_blks != 1) {
        cudaMalloc((void**)&tmp_arr, tmp_size * sizeof(T));
    }

    unsigned int offset = 0;
    while(summary.red_type == LargeSgm && summary.num_sgms_blks > 1) {
        // treat large segment case
        // Note that it enters here rarely, only for very large segments!
        unsigned int sh_mem_size = summary.block_size * sizeof(T);
        unsigned int num_blocks  = num_sgms * summary.num_sgms_blks;
        out_arr = tmp_arr + offset;
        if( (offset+num_blocks) > (tmp_size/2) ) {
            printf("EXITING: ALLOCATION ERROR!!!\n");
            exit(0);
        }
        offset += num_blocks;

#ifdef DEBUG_TIMING
        struct timeval t_start, t_end, t_diff;
        unsigned long int elapsed;
        gettimeofday(&t_start, NULL);
#endif
        
        sgmRedLargeKernel<OP><<< num_blocks, summary.block_size, sh_mem_size >>>
            (inp_arr, out_arr, sgm_size, summary.chunk_size, summary.num_sgms_blks);

#ifdef DEBUG_TIMING
        cudaThreadSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("INTERMEDIATE Parallel LARGE Sgm-Regular Reduce, array size: %d runs in: %lu microsecs, num_blocks: %d\n", num_sgms*sgm_size, elapsed, num_blocks);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("CUDA ERROR LARGE: %s\n", cudaGetErrorString(err));
#endif

        inp_arr  = out_arr;
        sgm_size = summary.num_sgms_blks;
        summary = findRedKind(num_sgms, sgm_size);
    }


    // FINAL STEP: ONE OF SeqSgm, SmallSgm, LargeSgm!
    if        ( summary.red_type == SeqSgm ) {
        
#ifdef DEBUG_TIMING
        cudaError_t err;
        struct timeval t_start, t_end, t_diff;
        unsigned long int elapsed;
        gettimeofday(&t_start, NULL);
#endif

        cudaMalloc((void**)&trnsp_arr, num_elems * sizeof(T));
        transposePad<T, TILE_SIZE>( inp_arr, trnsp_arr, num_sgms, 
                                    sgm_size, num_elems, OP::identity() );

//        err = cudaGetLastError();
//        if (err != cudaSuccess) printf("CUDA ERROR SEQUENT: %s\n", cudaGetErrorString(err));


        unsigned int num_blocks  = ceil( ((float)num_sgms) / summary.block_size);
        sequentialRedKernel<OP><<< num_blocks, summary.block_size >>>
            (trnsp_arr, d_out, num_sgms, sgm_size);
       
#ifdef DEBUG_TIMING
        cudaThreadSynchronize();
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("FINAL Parallel SEQUENTIAL Sgm-Regular Reduce, array size: %d runs in: %lu microsecs, num_blocks: %d\n", num_sgms*sgm_size, elapsed, num_blocks);

        err = cudaGetLastError();
        if (err != cudaSuccess) printf("CUDA ERROR SEQUENT: %s\n", cudaGetErrorString(err));
#endif
 
    } else if ( summary.red_type == SmallSgm ) {

        unsigned int sh_mem_size = summary.block_size * (sizeof(T) + sizeof(int));
        unsigned int num_blocks  = ceil( ((float)num_sgms) / summary.num_sgms_blks);

#ifdef DEBUG_TIMING
        struct timeval t_start, t_end, t_diff;
        unsigned long int elapsed;
        gettimeofday(&t_start, NULL);
#endif

        sgmRedSmallKernel<OP><<< num_blocks, summary.block_size, sh_mem_size >>>
            (inp_arr, d_out, num_sgms, sgm_size, summary.num_sgms_blks);
        cudaThreadSynchronize();

#ifdef DEBUG_TIMING
        cudaThreadSynchronize();
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("FINAL Parallel Small Sgm-Regular Reduce, array size: %d runs in: %lu microsecs, num_blocks: %d, sgm_size: %d\n",
                 num_sgms*sgm_size, elapsed, num_blocks, sgm_size);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("CUDA ERROR SEQUENT: %s\n", cudaGetErrorString(err));
#endif

    } else { // LargeSgm
        unsigned int sh_mem_size = summary.block_size * sizeof(T);
        unsigned int num_blocks  = num_sgms * summary.num_sgms_blks;

#ifdef DEBUG_TIMING
        struct timeval t_start, t_end, t_diff;
        unsigned long int elapsed;
        gettimeofday(&t_start, NULL);
#endif
        
        sgmRedLargeKernel<OP><<< num_blocks, summary.block_size, sh_mem_size >>>
            (inp_arr, d_out, sgm_size, summary.chunk_size, summary.num_sgms_blks);

#ifdef DEBUG_TIMING
        cudaThreadSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("FINAL Parallel LARGE Sgm-Regular Reduce, array size: %d runs in: %lu microsecs, num_blocks: %d\n", num_sgms*sgm_size, elapsed, num_blocks);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("CUDA ERROR LARGE: %s\n", cudaGetErrorString(err));
#endif
    }
    cudaThreadSynchronize();
    cudaFree(tmp_arr);
    cudaFree(trnsp_arr);
}
#endif //PAR_BB_HOST
