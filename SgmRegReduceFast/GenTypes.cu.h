#ifndef GEN_TYPES
#define GEN_TYPES

template<class T>
class Add {
  public:
    typedef T  ElmType;
    typedef T* ArrType;
    static __device__ __host__ inline T identity()                    { return (T)0.0;      }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2;   }
    static __device__ __host__ inline T elSize  ()                    { return sizeof(T); }
};

#endif // GEN_TYPES
