#include <bits/stdc++.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <chrono>
#include <math.h>
#define BLOCK_Size 1024

using namespace std;

typedef float precision;

float signum(double x)
{
    if (x > 0) return 1.0;
    if (x < 0) return -1.0;
    return 0.0;
}
__global__ void trace(precision *arr, int n, precision *dev_trace) 
{
    __shared__ precision sdata[BLOCK_Size];
    unsigned int tid = threadIdx.x;
    unsigned int acttid = blockDim.x* blockIdx.x + threadIdx.x;
    if(acttid<n)
    sdata[tid] = arr[acttid*n+acttid];
    else
    sdata[tid] = 0;
    __syncthreads();
    
    for(unsigned int s=1; s < BLOCK_Size; s *= 2) 
    {
        if (tid % (2*s) == 0) {
        sdata[tid] += sdata[tid + s];
        }
    __syncthreads();
    }

    if (tid == 0) dev_trace[blockIdx.x] = sdata[0];
}

__global__ void sec_trace(precision *arr, int n) 
{
    __shared__ precision sdata[1024];
    unsigned int tid = threadIdx.x;
    if (tid<n)
    sdata[tid] = arr[tid];
    else
    sdata[tid] = 0;
    __syncthreads();
    
    for(unsigned int s=1; s < 1024; s *= 2) 
    {
        if (tid % (2*s) == 0) {
        sdata[tid] += sdata[tid + s];
        }
    __syncthreads();
    }
    
    if (tid == 0) arr[blockIdx.x] = sdata[0];
}

precision CPU_trace(precision *X, precision *dev_trace, int size, int nblocks)
{
    precision *traces = (precision*)malloc(sizeof(precision));
    trace<<<nblocks,BLOCK_Size>>>(X,size,dev_trace);
    if (nblocks!=1)
    {
        sec_trace<<<1,1024>>>(dev_trace,nblocks);
    }
    cudaMemcpy(traces, &dev_trace[0], sizeof(precision), cudaMemcpyDeviceToHost);
    return traces[0];
}

precision* generate_H(int n)
{
    precision *H = (precision *)malloc(n * n * sizeof(precision));
    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            H[i * n + j] = exp(-.5*abs(i-j))*sin(i+1);
            H[j * n + i] = H[i * n + j];
        }
    }
    return H;
}


void dual_half(cublasHandle_t handle,precision* X_new,precision* X,int n)
{
    precision alpha = 1.0;
    precision beta = 0.0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, X_new, n, X_new, n, &beta, X, n);
}

void print_mat(precision * mat, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout<<mat[i*n +j]<<" ";
        }
        cout<<endl;
    }
}

precision *identity_mat(int n)
{
    precision *mat = (precision *)malloc(n * n * sizeof(precision));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                mat[i * n + j] = 1;
            }
            else
            {
                mat[i * n + j] = 0;
            }
        }
    }
    return mat;
}

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);  // Hamiltonian size
    int refine = atoi(argv[2]);
    int tensor = atoi(argv[3]);
    size_t bytes = n * n * sizeof(precision);
    int nblocks = (n % BLOCK_Size == 0 ? n / BLOCK_Size : (n / BLOCK_Size + 1));
    //INITIALIZE CPU
    precision alpha = 1.0;
    precision zero = 0.0;
    precision eps = 1e-4;                       // Small value such that +/- eps is finite in single precision
    precision Csp2 = 4.5;                       // Convergence criterion as derived in the Appendix
    precision sgn = 0.0;                          // Initial value of sgn
    precision Nocc = 20;                         // Occupation number
    precision *Identity = identity_mat(n);                      // Identity matrix
    precision *H = generate_H(n);
    int iteration = 0, maxlayer = 500;                     // Maximum number of layers

    precision h1 = -1.867;                       // Spectral lower bound
    precision hN = 1.867;                       // Spectral upper bound
    precision W0 = -1/(hN-h1);                  // Initial weight (scalar)
    precision B0 = (hN/(hN-h1));                // Initial Bias (diagonal matrix) 

    //INITIALIZE GPU
    precision *X, *X_new, *B, *I , *dev_trace;
    cudaMalloc(&X, bytes);
    cudaMemcpy(X, H, bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&dev_trace, nblocks * sizeof(precision));

    cudaMalloc(&I, bytes);
    cudaMemcpy(I, Identity, bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&X_new, bytes);

    cudaMalloc(&B, bytes);

    precision *v_sgn; // Keeps track of binary in-place learning choices
    v_sgn = (precision *)malloc(maxlayer * sizeof(precision));
    precision *idemp_err; // Local error estimate
    idemp_err = (precision *)malloc(maxlayer * sizeof(precision));

    cublasHandle_t handle;
    cublasCreate(&handle);

    if(tensor == 1)
    {
        cout<<"Used Tensor Core"<<endl;
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    }

    else
    {
        cout<<"Normal Operation"<<endl;
        cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    }
    
    cudaDeviceSynchronize();
    auto start = chrono::high_resolution_clock::now();

    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &B0, I, n, &zero, I, n, B, n);
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &W0, X, n, &B0, I, n, X_new, n);

    precision traceS = CPU_trace(X_new, dev_trace, n, nblocks),traceX;
    cout<<traceS<<endl;
    while(iteration<maxlayer)
    {
        dual_half(handle,X_new,X,n);
        traceX = CPU_trace(X, dev_trace, n, nblocks);
        idemp_err[iteration] = traceS-traceX; 
        // cout<<idemp_err[iteration]<<std::scientific<<endl;
        sgn = signum(abs(2*traceS - traceX - Nocc) - abs(traceX - Nocc) - sgn*eps); 
        v_sgn[iteration] = sgn;                          
        W0 = sgn;
        B0 = (1-sgn);
        
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &B0, X_new, n, &zero, I, n, B, n);
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &W0, X, n, &alpha, B, n, X_new, n);

        traceS = W0*traceX + (1-sgn)*traceS;        

        if (idemp_err[iteration] <= 0)        
        break;
        if (iteration > 1 && v_sgn[iteration-1] != v_sgn[iteration-2] && 
                idemp_err[iteration] >= Csp2*idemp_err[iteration-2]*idemp_err[iteration-2])            
        break;
        iteration++;
    }

    if (refine == 1)
    {
        cout<<"Refinement Step"<<endl;
        precision minus = -1.0;
        precision two = 2.0;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, X_new, n, X_new, n, &zero, X, n);
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &minus, X, n, &two, X_new, n, X, n); 
        traceS = CPU_trace(X, dev_trace, n, nblocks);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, X, n, X, n, &zero, X_new, n);
        traceX = CPU_trace(X_new, dev_trace, n, nblocks);
        cout<<traceS-traceX<<endl;
    }
    else
        cout<<"No Refinement Step"<<endl;


    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    precision time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-9;
    cout << "Time taken by program is : " << fixed << time_taken << setprecision(9) << " sec" << endl;
    cout<<iteration<<endl;
    return 0;
}