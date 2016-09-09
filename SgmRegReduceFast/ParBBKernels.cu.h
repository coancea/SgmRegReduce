#ifndef UTIL_KERNELS
#define UTIL_KERNELS 

#include "GenTypes.cu.h"

/************************/
/*** TRANSPOSE Kernel ***/
/************************/
// blockDim.y = TILE; blockDim.x = TILE
// each block transposes a square TILE
template <class T, int TILE> 
__global__ void matTransposeTiledPadKer(T* A, T* B, int heightA, int widthA, int orig_size, T padel) {

  __shared__ T tile[TILE][TILE+1];

  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;

  int ind = y*widthA+x;
  if( x < widthA && y < heightA )
      tile[threadIdx.y][threadIdx.x] = A[ind]; //(ind < orig_size) ? A[ind] : padel;

  __syncthreads();

  x = blockIdx.y * TILE + threadIdx.x; 
  y = blockIdx.x * TILE + threadIdx.y;

  ind = y*heightA + x;
  if( x < heightA && y < widthA )
      B[ind] = tile[threadIdx.x][threadIdx.y];
}

/***************************************/
/*** Scan Inclusive Helpers & Kernel ***/
/***************************************/
template<class OP>
__device__ inline
typename OP::ElmType scanIncWarp(   volatile typename OP::ElmType* ptr, 
                                    const unsigned int idx 
) {
    typedef typename OP::ElmType T;
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  ptr[idx] = OP::apply(ptr[idx-1],  ptr[idx]);
    if (lane >= 2)  ptr[idx] = OP::apply(ptr[idx-2],  ptr[idx]);
    if (lane >= 4)  ptr[idx] = OP::apply(ptr[idx-4],  ptr[idx]);
    if (lane >= 8)  ptr[idx] = OP::apply(ptr[idx-8],  ptr[idx]);
    if (lane >= 16) ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]);
    return const_cast<T&>(ptr[idx]);
}

template<class OP>
__device__ inline
typename OP::ElmType scanIncBlock(  volatile typename OP::ElmType* ptr, 
                                    const unsigned int idx
) {
    typedef typename OP::ElmType T;
    const unsigned int lane   = idx &  31;
    const unsigned int warpid = idx >> 5;

    T val = scanIncWarp<OP>(ptr,idx);
    __syncthreads();

    // place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    if (lane == 31) { ptr[warpid] = const_cast<T&>(ptr[idx]); }
    __syncthreads();

    //
    if (warpid == 0) scanIncWarp<OP>(ptr, idx);
    __syncthreads();

    if (warpid > 0) {
        val = OP::apply(ptr[warpid-1], val);
    }

    return val;
}
#if 0
template<class OP, class T>
__global__ void 
scanIncKernel(T* d_in, T* d_out, unsigned int d_size) {
    extern __shared__ char sh_mem1[];
    volatile T* sh_memT = (volatile T*)sh_mem1;
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + tid;
    T el    = (gid < d_size) ? d_in[gid] : OP::identity();
    sh_memT[tid] = el;
    __syncthreads();
    T res   = scanIncBlock < OP, T >(sh_memT, tid);
    if (gid < d_size) d_out [gid] = res; 
}
#endif

/*************************************************/
/*************************************************/
/*** Segmented Inclusive Scan Helpers & Kernel ***/
/*************************************************/
/*************************************************/
template<class OP, class F>
__device__ inline
typename OP::ElmType sgmScanIncWarp(    volatile typename OP::ElmType* ptr, 
                                        volatile F* flg, const unsigned int idx
) {
    typedef typename OP::ElmType T;
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-1], ptr[idx]); }
        flg[idx] = flg[idx-1] | flg[idx];
    }
    if (lane >= 2)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-2], ptr[idx]); }
        flg[idx] = flg[idx-2] | flg[idx];
    }
    if (lane >= 4)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-4], ptr[idx]); }
        flg[idx] = flg[idx-4] | flg[idx];
    }
    if (lane >= 8)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-8], ptr[idx]); }
        flg[idx] = flg[idx-8] | flg[idx];
    }
    if (lane >= 16)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]); }
        flg[idx] = flg[idx-16] | flg[idx];
    }

    return const_cast<T&>(ptr[idx]);
}

template<class OP, class F>
__device__ inline
typename OP::ElmType sgmScanIncBlock(   volatile typename OP::ElmType* ptr, 
                                        volatile F* flg, const unsigned int idx
) {
    typedef typename OP::ElmType T;
    const unsigned int lane   = idx &  31;
    const unsigned int warpid = idx >> 5;
    const unsigned int warplst= (warpid<<5) + 31;

    // 1a: record whether this warp begins with an ``open'' segment.
    bool warp_is_open = (flg[(warpid << 5)] == 0);
    __syncthreads();

    // 1b: intra-warp segmented scan for each warp
    T val = sgmScanIncWarp<OP,F>(ptr,flg,idx);

    // 2a: the last value is the correct partial result
    T warp_total = const_cast<T&>(ptr[warplst]);
    
    // 2b: warp_flag is the OR-reduction of the flags 
    //     in a warp, and is computed indirectly from
    //     the mindex in hd[]
    bool warp_flag = flg[warplst]!=0 || !warp_is_open;
    bool will_accum= warp_is_open && (flg[idx] == 0);

    __syncthreads();

    // 2c: the last thread in a warp writes partial results
    //     in the first warp. Note that all fit in the first
    //     warp because warp = 32 and max block size is 32^2
    if (lane == 31) {
        ptr[warpid] = warp_total; //ptr[idx]; 
        flg[warpid] = warp_flag;
    }
    __syncthreads();

    // 
    if (warpid == 0) sgmScanIncWarp<OP,F>(ptr, flg, idx);
    __syncthreads();

    if (warpid > 0 && will_accum) {
        val = OP::apply(ptr[warpid-1], val);
    }
    return val;
}
#if 0
template<class OP>
__global__ void 
sgmScanIncKernel(   typename OP::ElmType* d_in 
                ,   int* flags, 
                ,   typename OP::ElmType* d_out
                ,   int* f_rec, 
                ,   typename T* d_rec, unsigned int d_size) {
    extern __shared__ char sh_mem[];
    volatile T*   vals_sh = (volatile T*)sh_mem;
    volatile int* flag_sh = (int*) (vals_sh + blockDim.x);
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + tid;
    int fl;   
    if (gid < d_size) { vals_sh[tid] = d_in[gid];      fl = flags[gid]; }
    else              { vals_sh[tid] = OP::identity(); fl = 0;          }
    flag_sh[tid] = fl;
    __syncthreads();
    T res = sgmScanIncBlock <OP, T>(vals_sh, flag_sh, tid);
    if (gid < d_size) d_out [gid] = res; 

    // set the flags and data for the recursive step!
    if(tid == 0)  { f_rec[blockIdx.x] = 0; }
    __syncthreads();
    if(fl  >  0)  { f_rec[blockIdx.x] = 1; }
    if(tid == (blockDim.x - 1)) { d_rec[blockIdx.x] = res; }
}
#endif
///////////////////////////////////////////////////////

template<class OP>
__global__ void 
sgmRedLargeKernel(  typename OP::ElmType* inp_arr
                 ,  typename OP::ElmType* out_arr 
                 ,  const unsigned int    sgm_size
                 ,  const unsigned int    chunk_size
                 ,  const unsigned int    blks_sgm
) {
    typedef typename OP::ElmType T;
    extern __shared__ char sh_mem1[];
    volatile T* sh_memT = (volatile T*)sh_mem1;
    
    T el = OP::identity();

    unsigned int beg_sgm = (blockIdx.x / blks_sgm) * sgm_size;
    unsigned int in_sgm  = (blockIdx.x % blks_sgm) * blockDim.x * chunk_size + // intra-segment offset
                            threadIdx.x;

    for(unsigned int i = 0; 
        i < chunk_size && in_sgm < sgm_size; 
        i++, in_sgm += blockDim.x
    ) {
        T cur_el = inp_arr[in_sgm+beg_sgm];
        el = OP::apply(el, cur_el);
    }

    sh_memT[threadIdx.x] = el;
    __syncthreads();
    T res  = scanIncBlock<OP>(sh_memT, threadIdx.x);

    // write result
    if (threadIdx.x == (blockDim.x-1)) {
        out_arr[blockIdx.x] = res;
    }  
}

template<class OP>
__global__ void 
sgmRedSmallKernel(  typename OP::ElmType* inp_arr
                 ,  typename OP::ElmType* out_arr 
                 ,  const unsigned int    num_sgms
                 ,  const unsigned int    sgm_size
                 ,  const unsigned int    sgms_blk
) {
    typedef typename OP::ElmType T;
    extern __shared__ char sh_mem[];
    volatile T*   vals_sh = (volatile T*) sh_mem;
    volatile int* flag_sh = (volatile int*) (vals_sh + blockDim.x);

    T el;
    unsigned int gid = blockIdx.x*sgms_blk*sgm_size + threadIdx.x;
    if( ( threadIdx.x < (sgms_blk*sgm_size) ) &&  
        ( gid < (num_sgms*sgm_size)           )   ) {        
        el = inp_arr[gid];
    } else {
        el = OP::identity();
    }

    vals_sh[threadIdx.x] = el;
    flag_sh[threadIdx.x] = ( (threadIdx.x % sgm_size) == 0 ) ? 1 : 0;

    __syncthreads();
    el  = sgmScanIncBlock <OP, int> (vals_sh, flag_sh, threadIdx.x);
    __syncthreads();
    vals_sh[threadIdx.x] = el;
    __syncthreads();

    gid = blockIdx.x*sgms_blk + threadIdx.x;
    if( threadIdx.x < sgms_blk && gid < num_sgms ) {
        out_arr[gid] = vals_sh[(threadIdx.x+1)*sgm_size - 1];
    }
}


template<class OP>
__global__ void 
sequentialRedKernel(  typename OP::ElmType* inp_arr
                   ,  typename OP::ElmType* out_arr 
                   ,  const unsigned int    num_sgms
                   ,  const unsigned int    sgm_size
) {
    typedef typename OP::ElmType T;

    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < num_sgms) {
        T acc = OP::identity();
        unsigned int num_elems = num_sgms * sgm_size;
        for( ; gid < num_elems; gid += num_sgms) {
            acc = OP::apply(acc, inp_arr[gid]);
        }
        out_arr[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    }
}

#endif //UTIL_KERNELS

