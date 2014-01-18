#include <stdio.h>
#include <assert.h>
#include <cuda.h>

__global__ void matmul_kernel( unsigned len, float* a , float* b , float* c){

	//
	//  THIS IS THE SIMPLE WAY TO DO IT, NOT THE ***FAST WAY***
	//

	// get index in c
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//return if over the length
	if(row>=len | col>=len){return;}

	// initialize local variable to hold values while the sum is done
	float sum = 0;
	unsigned j;

	// scan the row of a, the col of b
	for(j=0;j<len;j++){
		sum +=  a[ row * len + j ] * b[ j * len + col ];
	}

	// write final value into output array
	c[ len * row + col  ] = sum;

}

__global__ void matmul_kernel_shared( unsigned len, float* a , float* b , float* c){

	// take advantage of data resue w shared memory
	// this method might not be scalable though since there is only 16kb shared mem and usage scales as 2*len*blockDim!

	// get index in c
	int offset_x = blockIdx.x * blockDim.x;
	int offset_y = blockIdx.y * blockDim.y;
	int row = offset_y + threadIdx.y;
	int col = offset_x + threadIdx.x;

	//return if over the length
	if(row>=len | col>=len){return;}

	// initialize local variable to hold values while the sum is done
	float sum = 0;
	unsigned j,k;

	// declare shared memory
	extern __shared__ float sub_a[];
	float* sub_b = &sub_a[blockDim.x*len];

	//have 0,0 thread load in data to shared, recast both into row-major
	if( blockIdx.x+blockIdx.y == 0){
		// load a
		for(     j = 0 ; j < min( len - offset_y , blockDim.y ) ; j++){  //j is column
			for( k = 0 ; k < len ; k++){ // k goes across the row 
				sub_a[ len * j + k ] = a[ (j+offset_y) * len + k ];
			}
		}
		// load b
		for(     j = 0 ; j < min( len - offset_x , blockDim.x ) ; j++){  // j is column
			for( k = 0 ; k < len ; k++){ // k goes down the colum
				sub_b[ len * j + k ] = b[ k * len + (j+offset_y) ];
			}
		}
	}
	// sync, other threads need to wait for data
	__syncthreads();

	// scan the row of a, the col of b using shared mem.
	for(j=0;j<len;j++){
		sum +=  sub_a[ (row-offset_y) * len + j ] * sub_b[ (col-offset_x) * len + j  ];
	}

	// write final value into output array
	c[ len * row + col  ] = sum;

}

int main(){

	// declare
	float* 		a;
	float*  	b;  
	float* 		c;
	float* 		d_a;
	float*		d_b;
	float*		d_c;
	unsigned  	len_a, len_b, j, k;
	unsigned 	bytes_a, bytes_b, bytes_c;
	dim3 		NUM_THREADS, blks;

	//open files, get lengths, make sure they are equal
	FILE* af = fopen("a","r");
	FILE* bf = fopen("b","r");
	FILE* cf;

	fscanf(af,"%u\n",&len_a);
	fscanf(bf,"%u\n",&len_b);
	printf("dims a,b = %u , %u\n",len_a,len_b);
	assert(len_a==len_b);
	bytes_a = len_a * len_a * sizeof(float);
	bytes_b = len_b * len_b * sizeof(float);
	bytes_c = len_b * len_b * sizeof(float);

	//allocate arrays
	a = (float*) malloc( bytes_a );
	b = (float*) malloc( bytes_b );
	c = (float*) malloc( bytes_b );

	//allocate device arrays
	cudaMalloc( &d_a , bytes_a );  //must be pointer to the point, since the actual point value is being changed, not the value it points to
	cudaMalloc( &d_b , bytes_b );
	cudaMalloc( &d_c , bytes_c );

	// read in data
	for(j=0;j<len_a;j++){
		for(k=0;k<len_a;k++){
			fscanf(af,"%E \n",&a[j*len_a+k]);  //row major
			fscanf(bf,"%E \n",&b[j*len_a+k]);
		}
	}

	// close files
	fclose(af); fclose(bf);

	// copy data to device
	cudaMemcpy( d_a , a , bytes_a , cudaMemcpyHostToDevice );
	cudaMemcpy( d_b , b , bytes_b , cudaMemcpyHostToDevice );
	
	//calculate the number of blocks from the number of threads
	//NUM_THREADS.x = NUM_THREADS.y = 16;
	//blks.x = blks.y = (len_a + NUM_THREADS.x - 1 ) / NUM_THREADS.x;
	//NUM_THREADS.z = blks.z = 1;
	//printf("NUM_THREADS(%4u,%4u,   0)\n       blks(%4u,%4u,   0)\n",NUM_THREADS.x,NUM_THREADS.y,blks.x,blks.y);
	// launch kernel to do a*b=c
	//matmul_kernel <<< blks, NUM_THREADS >>> (len_a , d_a , d_b , d_c);
	// launch kernel for shared memory implementation
	//calculate the number of blocks from the number of threads
	NUM_THREADS.x   = NUM_THREADS.y = 2;
	blks.x = blks.y = (len_a + NUM_THREADS.x - 1 ) / NUM_THREADS.x;
	NUM_THREADS.z   = blks.z = 1;
	unsigned shared_mem_size = 2*len_a*NUM_THREADS.x*sizeof(float);
	printf("NUM_THREADS(%4u,%4u,   0)\n       blks(%4u,%4u,   0)\n",NUM_THREADS.x,NUM_THREADS.y,blks.x,blks.y);
	printf("shared_mem_size = %u\n",shared_mem_size);
	matmul_kernel_shared <<< blks, NUM_THREADS , shared_mem_size >>> (len_a , d_a , d_b , d_c);

	// check for errors
	if(cudaPeekAtLastError()){
		printf("CUDA ERROR, %s\n",cudaGetErrorString(cudaPeekAtLastError()));
		return 1;
	}

	//copy c back
	cudaMemcpy( c , d_c , bytes_b , cudaMemcpyDeviceToHost );

	// write a,b,c to files in matrix format to be read by matlab for plotting
	af = fopen("a_out","w");
	bf = fopen("b_out","w");
	cf = fopen("c_out","w");
	for(j=0;j<len_a;j++){
		for(k=0;k<len_a;k++){
			fprintf(af,"%10.8E ",a[j*len_a+k]);  //row major
			fprintf(bf,"%10.8E ",b[j*len_a+k]);
			fprintf(cf,"%10.8E ",c[j*len_a+k]);
		}
		fprintf(af,"\n"); 
		fprintf(bf,"\n");
		fprintf(cf,"\n");
	}
	fclose(af);
	fclose(bf);
	fclose(cf);

	// return zero if all ok
	return 0;

}