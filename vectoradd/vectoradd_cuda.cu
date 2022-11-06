/*
Vector addition code

Copyright 2022 Martin Burtscher

Redistribution in source or binary form, with or without modification, is not
permitted. Use in source or binary form, with or without modification, is only
permitted for academic use in CS 4380 and CS 5351 at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <sys/time.h>


static const int ThreadsPerBlock = 512;


static __global__ void vadd(const int* const a, const int* const b, int* const c, const int elems)
{
  // perform vector addition
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < elems) {
    c[i] = a[i] + b[i];
  }
}


static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}


int main(int argc, char* argv [])
{
  printf("Vector addition v1.6\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s vector_elements\n", argv[0]); exit(-1);}
  const int elems = atoi(argv[1]);
  if (elems < 1) {fprintf(stderr, "ERROR: vector_elements must be at least 1\n"); exit(-1);}
  printf("vector elements: %d\n", elems);

  // allocate vectors
  int* const a = new int [elems];
  int* const b = new int [elems];
  int* const c = new int [elems];

  // initialize vectors
  for (int i = 0; i < elems; i++) a[i] = i;
  for (int i = 0; i < elems; i++) b[i] = elems - i;
  for (int i = 0; i < elems; i++) c[i] = -1;

  // allocate vectors on GPU
  int* d_a;
  int* d_b;
  int* d_c;
  cudaMalloc((void **)&d_a, sizeof(int) * elems);
  CheckCuda(__LINE__);
  cudaMalloc((void **)&d_b, sizeof(int) * elems);
  CheckCuda(__LINE__);
  cudaMalloc((void **)&d_c, sizeof(int) * elems);
  CheckCuda(__LINE__);

  // initialize vectors on GPU
  cudaMemcpy(d_a, a, sizeof(int) * elems, cudaMemcpyHostToDevice);
  CheckCuda(__LINE__);
  cudaMemcpy(d_b, b, sizeof(int) * elems, cudaMemcpyHostToDevice);
  CheckCuda(__LINE__);
  cudaMemcpy(d_c, c, sizeof(int) * elems, cudaMemcpyHostToDevice);  // optional
  CheckCuda(__LINE__);

  // start time
  timeval beg, end;
  gettimeofday(&beg, NULL);

  // execute timed code
  vadd<<<(elems + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_a, d_b, d_c, elems);
  cudaDeviceSynchronize();  // wait for kernel to finish

  // end time
  gettimeofday(&end, NULL);
  CheckCuda(__LINE__);
  const double runtime = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

  // get result from GPU
  cudaMemcpy(c, d_c, sizeof(int) * elems, cudaMemcpyDeviceToHost);
  CheckCuda(__LINE__);

  // verify result
  for (int i = 0; i < elems; i++) {
    if (c[i] != elems) {fprintf(stderr, "ERROR: incorrect result\n"); exit(-1);}
  }
  printf("verification passed\n");

  // clean up
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete [] a;
  delete [] b;
  delete [] c;
  return 0;
}
