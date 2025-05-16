#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>

static const int cmp_tblock_size = 32; // Fixed to 32, cannot be modified.
static const int cmp_chunk = 1024;

struct PrivateTimingGPU {
    cudaEvent_t start;
    cudaEvent_t stop;
};
class TimingGPU
{
    private:
        PrivateTimingGPU *privateTimingGPU;

    public:
        TimingGPU();
        ~TimingGPU();
        void StartCounter();
        void StartCounterFlags();
        float GetCounter();

};


TimingGPU::TimingGPU() { privateTimingGPU = new PrivateTimingGPU;  }
TimingGPU::~TimingGPU() { }

/** ************************************************************************
 * @brief           Start timer.
 * *********************************************************************** */
void TimingGPU::StartCounter()
{
    cudaEventCreate(&((*privateTimingGPU).start));
    cudaEventCreate(&((*privateTimingGPU).stop));
    cudaEventRecord((*privateTimingGPU).start,0);
}

/** ************************************************************************
 * @brief           Start timer with flags.
 * *********************************************************************** */
void TimingGPU::StartCounterFlags()
{
    int eventflags = cudaEventBlockingSync;

    cudaEventCreateWithFlags(&((*privateTimingGPU).start),eventflags);
    cudaEventCreateWithFlags(&((*privateTimingGPU).stop),eventflags);
    cudaEventRecord((*privateTimingGPU).start,0);
}

/** ************************************************************************
 * @brief           End timer, get count in ms.
 * *********************************************************************** */
float TimingGPU::GetCounter()
{
    float time;
    cudaEventRecord((*privateTimingGPU).stop, 0);
    cudaEventSynchronize((*privateTimingGPU).stop);
    cudaEventElapsedTime(&time,(*privateTimingGPU).start,(*privateTimingGPU).stop);
    return time;
}

typedef enum {
    CUSZP_MODE_PLAIN   = 0, // Plain   fixed-length encoding mode
} cuszp_mode_t;

typedef enum {
    CUSZP_TYPE_FLOAT  = 0,  // Single precision floating point (f32)
} cuszp_type_t;



__device__ inline int quantization(float data, float recipPrecision)
{
    int result;
    asm("{\n\t"
        ".reg .f32 dataRecip;\n\t"
        ".reg .f32 temp1;\n\t"
        ".reg .s32 s;\n\t"
        ".reg .pred p;\n\t"
        "mul.f32 dataRecip, %1, %2;\n\t"
        "setp.ge.f32 p, dataRecip, -0.5;\n\t"
        "selp.s32 s, 0, 1, p;\n\t"
        "add.f32 temp1, dataRecip, 0.5;\n\t"
        "cvt.rzi.s32.f32 %0, temp1;\n\t"
        "sub.s32 %0, %0, s;\n\t"
        "}": "=r"(result) : "f"(data), "f"(recipPrecision)
    );
    return result;
}
__device__ inline int get_bit_num(unsigned int x)
{
    int leading_zeros;
    asm("clz.b32 %0, %1;" : "=r"(leading_zeros) : "r"(x));
    return 32 - leading_zeros;
}
// CUDA Kernel for compression
__global__ void cuSZp_compress_kernel_plain_f32(const float* const __restrict__ oriData, 
                                                unsigned char* const __restrict__ cmpData, 
                                                volatile unsigned int* const __restrict__ cmpOffset, 
                                                volatile unsigned int* const __restrict__ locOffset, 
                                                volatile int* const __restrict__ flag, 
                                                const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = 1; // Fixed for simplicity, adjust if needed
    const int rate_ofs = (nbEle + 1024 - 1) / 1024 * 1024 / 32; // Adjust block size if needed
    const float recipPrecision = 0.5f / eb;

//    if(eb < FLT_MIN)
//    	  { printf("eb is very low, lower than FLT_MIN:  %f\n", eb); }

//    printf("eb is :  %f\n", eb); 
      

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    int block_idx;
    int currQuant, lorenQuant, prevQuant, maxQuant;
    int absQuant[32]; // Adjust array size if needed
    unsigned int sign_flag[block_num];
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;
    float4 tmp_buffer;
    uchar4 tmp_char;

    base_start_idx = warp * 32; // Adjust if needed
    for(int j = 0; j < block_num; j++)
    {
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        sign_flag[j] = 0;
        block_idx = base_block_start_idx / 32;
        prevQuant = 0;
        maxQuant = 0;

        if(base_block_end_idx < nbEle)
        {
            #pragma unroll 8
            for(int i = base_block_start_idx; i < base_block_end_idx; i += 4)
            {
                tmp_buffer = reinterpret_cast<const float4*>(oriData)[i / 4];
                quant_chunk_idx = j * 32 + i % 32;

                currQuant = quantization(tmp_buffer.x, recipPrecision);
                lorenQuant = currQuant - prevQuant; //1D lorenzo
                prevQuant = currQuant;
                sign_ofs = i % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs); //seprate the sign bits
                absQuant[quant_chunk_idx] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];

                currQuant = quantization(tmp_buffer.y, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i + 1) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx + 1] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx + 1] ? maxQuant : absQuant[quant_chunk_idx + 1];

                currQuant = quantization(tmp_buffer.z, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i + 2) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx + 2] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx + 2] ? maxQuant : absQuant[quant_chunk_idx + 2];

                currQuant = quantization(tmp_buffer.w, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i + 3) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx + 3] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx + 3] ? maxQuant : absQuant[quant_chunk_idx + 3];
            }
        }
        else
        {
            if(base_block_start_idx >= nbEle)
            {
                quant_chunk_idx = j * 32 + base_block_start_idx % 32;
                for(int i = quant_chunk_idx; i < quant_chunk_idx + 32; i++) absQuant[i] = 0;
            }
            else
            {
                int remainbEle = nbEle - base_block_start_idx;
                int zeronbEle = base_block_end_idx - nbEle;

                for(int i = base_block_start_idx; i < base_block_start_idx + remainbEle; i++)
                {
                    quant_chunk_idx = j * 32 + i % 32;
                    currQuant = quantization(oriData[i], recipPrecision);
                    lorenQuant = currQuant - prevQuant;
                    prevQuant = currQuant;
                    sign_ofs = i % 32;
                    sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                    absQuant[quant_chunk_idx] = abs(lorenQuant);
                    maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
                }

                quant_chunk_idx = j * 32 + nbEle % 32;
                for(int i = quant_chunk_idx; i < quant_chunk_idx + zeronbEle; i++) absQuant[i] = 0;
            }  
        }

        fixed_rate[j] = get_bit_num(maxQuant);
        thread_ofs += (fixed_rate[j]) ? (4 + fixed_rate[j] * 4) : 0;
        cmpData[block_idx] = (unsigned char)fixed_rate[j];
        __syncthreads();
    }

    #pragma unroll 5
    for(int i = 1; i < 32; i <<= 1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane == 31) 
    {
        locOffset[warp + 1] = thread_ofs;
        __threadfence();
        if(warp == 0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp + 1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp > 0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback > 0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status == 0);
                if(status == 2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status == 1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp > 0)
    {
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp == gridDim.x - 1) cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    for(int j = 0; j < block_num; j++)
    {
        int chunk_idx_start = j * 32;

        tmp_byte_ofs = (fixed_rate[j]) ? (4 + fixed_rate[j] * 4) : 0;
        #pragma unroll 5
        for(int i = 1; i < 32; i <<= 1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(fixed_rate[j])
        {
            tmp_char.x = 0xff & (sign_flag[j] >> 24);
            tmp_char.y = 0xff & (sign_flag[j] >> 16);
            tmp_char.z = 0xff & (sign_flag[j] >> 8);
            tmp_char.w = 0xff & sign_flag[j];
            reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs / 4] = tmp_char;
            cmp_byte_ofs += 4;

            int mask = 1;
            for(int i = 0; i < fixed_rate[j]; i++)
            {
                tmp_char.x = 0;
                tmp_char.y = 0;
                tmp_char.z = 0;
                tmp_char.w = 0;

                tmp_char.x = (((absQuant[chunk_idx_start + 0] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start + 1] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start + 2] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start + 3] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start + 4] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start + 5] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start + 6] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start + 7] & mask) >> i) << 0);

                tmp_char.y = (((absQuant[chunk_idx_start + 8] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start + 9] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start + 10] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start + 11] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start + 12] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start + 13] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start + 14] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start + 15] & mask) >> i) << 0);

                tmp_char.z = (((absQuant[chunk_idx_start + 16] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start + 17] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start + 18] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start + 19] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start + 20] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start + 21] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start + 22] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start + 23] & mask) >> i) << 0);
                
                tmp_char.w = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);

                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs / 4] = tmp_char;
                cmp_byte_ofs += 4;
                mask <<= 1;
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}

void cuSZp_compress_plain_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * cmp_chunk - 1) / (bsize * cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    unsigned int glob_sync;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_compress_kernel_plain_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}


void cuSZp_compress(void* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cuszp_type_t type, cuszp_mode_t mode, cudaStream_t stream)
{   
    if (type == CUSZP_TYPE_FLOAT) {
        if (mode == CUSZP_MODE_PLAIN) {
            cuSZp_compress_plain_f32((float*)d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
        }   
    } 
}





int main()
{
    // For measuring the end-to-end throughput.
    TimingGPU timer_GPU;

    // Input data preparation on CPU.
    float* oriData = NULL;
    
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 1024*1024*512; // 2 GB fp32 data.
    size_t cmpSize1 = 0;
    oriData = (float*)malloc(nbEle*sizeof(float));
    
    cmpBytes = (unsigned char*)malloc(nbEle*sizeof(float));

    // Initialize oriData.
    printf("Generating test data...\n\n");
    float startValue = -20.0f;
    float step = 0.1f;
    float endValue = 20.0f;
    size_t idx = 0;
    float value = startValue;
    while (idx < nbEle) 
    {
        oriData[idx++] = value;
        value += step;
        if (value > endValue)
        {
            value = startValue;
        }
    }

//--begin
srand(42);  // Fixed seed for reproducibility
printf("BEGIN: Generating diverse FP32 test data...\n\n");
for (size_t idx = 0; idx < nbEle; idx++) {
    float val;
    float p = rand() / (float)RAND_MAX;

    if (p < 0.01f) {
        // 1% chance: maximum positive normal
        val = FLT_MAX;
    } else if (p < 0.02f) {
        // 1% chance: maximum negative normal
        val = -FLT_MAX;
    } else if (p < 0.10f) {
        // 8% chance: subnormal positive
        val = FLT_TRUE_MIN * (rand() / (float)RAND_MAX) * 10.0f;
    } else if (p < 0.18f) {
        // 8% chance: subnormal negative
        val = -FLT_TRUE_MIN * (rand() / (float)RAND_MAX) * 10.0f;
    } else if (p < 0.20f) {
        // 2% chance: exact smallest subnormals
        val = (rand() % 2 == 0) ? FLT_TRUE_MIN : -FLT_TRUE_MIN;
    } else {
        // 80% chance: small normal values in [-1, 1]
        val = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    oriData[idx] = val;
}

//--end

    printf("END: Generating diverse FP32 test data...\n\n");

    // Get value range, making it a REL errMode test -- remove this will be ABS errMode.
    float max_val = oriData[0];
    float min_val = oriData[0];
    for(size_t i=0; i<nbEle; i++)
    {
        if(oriData[i]>max_val)
            max_val = oriData[i];
        else if(oriData[i]<min_val)
            min_val = oriData[i];
    }
    float errorBound = (max_val - min_val) * 1E-2f;

    // Input data preparation on GPU.
    float* d_oriData;
    float* d_decData;
    unsigned char* d_cmpBytes;
    cudaMalloc((void**)&d_oriData, sizeof(float)*nbEle);
    cudaMemcpy(d_oriData, oriData, sizeof(float)*nbEle, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_decData, sizeof(float)*nbEle);
    cudaMalloc((void**)&d_cmpBytes, sizeof(float)*nbEle);

    // Initializing CUDA Stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup for NVIDIA GPU.
    for(int i=0; i<3; i++)
    {
        cuSZp_compress_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize1, errorBound, stream);
    }

    // cuSZp-p testing.
    printf("=================================================\n");
    printf("=========Testing cuSZp-p-f32 on REL 1E-2=========\n");
    printf("=================================================\n");
    // cuSZp compression.
    timer_GPU.StartCounter(); // set timer
    cuSZp_compress_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize1, errorBound, stream);
    float cmpTime = timer_GPU.GetCounter();

    // Transfer compressed data to CPU then back to GPU, making sure compression ratio is correct.
    // No need to add this part for real-world usages, this is only for testing compresion ratio correctness.
    unsigned char* cmpBytes_dup1 = (unsigned char*)malloc(cmpSize1*sizeof(unsigned char));
    cudaMemcpy(cmpBytes_dup1, d_cmpBytes, cmpSize1*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemset(d_cmpBytes, 0, sizeof(float)*nbEle); // set to zero for double check.
    cudaMemcpy(d_cmpBytes, cmpBytes_dup1, cmpSize1*sizeof(unsigned char), cudaMemcpyHostToDevice);
        
    // Save compressed data to a .bin file
    FILE* fp = fopen("compressed_output.bin", "wb");
    if (fp == NULL) {
        std::cerr << "Failed to open file for writing.\n";
        exit(1);
    }
    fwrite(cmpBytes_dup1, sizeof(unsigned char), cmpSize1, fp);
    fclose(fp);
    printf("Compressed data written to 'test_data.cuszp.bin'\n");


    // Print result.
    printf("cuSZp-p finished!\n");
    printf("cuSZp-p compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/cmpTime);
    printf("cuSZp-p compression ratio: %f\n", (nbEle*sizeof(float)/1024.0/1024.0)/(cmpSize1*sizeof(unsigned char)/1024.0/1024.0));

    free(oriData);
    free(cmpBytes);
    cudaStreamDestroy(stream);

    return 0;
}