#include <stdio.h>
#include <assert.h>
#include <cuda.h>

/*
CUDA Tutorial, matrix-matrix multiply
UC Berkeley Reactor Design and Neutronics Group
Ryan M. Bergmann - 1/22/2014
*/

__global__ void matmul_kernel( unsigned len, float* a , float* b , float* c){

	//
	//  THIS IS THE SIMPLE WAY TO DO IT, NOT THE ***FAST WAY*** -> uses 2*N^3 global loads
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

	// take advantage of data resue w/ shared memory.  Uses tiles and loops over through them.  Global loads now 2N^3/(blockDim.x*blockDim.y)?
	// Programmed for, but **NOT TESTED** FOR cases where the block dimensions do not line up exactly with the matrix dimensions

	// get index in c
	int offset_x = blockIdx.x * blockDim.x;
	int offset_y = blockIdx.y * blockDim.y;	
	int row = offset_y + threadIdx.y;
	int col = offset_x + threadIdx.x;

	//return if over the length
	if(row>=len | col>=len){return;}

	// initialize local variable to hold values while the sum is done
	float sum = 0;
	unsigned j,g,sub_a_row,sub_a_col,sub_b_row,sub_b_col,sub_lim_x,sub_lim_y;
	unsigned n_blocks_x = ( len + blockDim.x - 1 ) / blockDim.x;
	unsigned n_blocks_y = ( len + blockDim.y - 1 ) / blockDim.y;

	// declare shared memory
	extern __shared__ float sub_a[];
	float* sub_b = &sub_a[blockDim.x*blockDim.y];

	//have 0,0 thread load in data to shared
	for(g=0 ; g < n_blocks_x ; g++){      // tile row

		// compute the global indicies of this submatrix
		sub_a_row =   offset_y; //const
		sub_a_col =   g * blockDim.x ;
		sub_b_row =   g * blockDim.y ;
		sub_b_col =   offset_x; //const

		// compute limits 
		sub_lim_x = min( len - sub_a_col , blockDim.x );
		sub_lim_y = min( len - sub_b_row , blockDim.y );

		// load shared memory
		if( threadIdx.x+threadIdx.y == 0){
			// load a row by row (saves programming another loop and dealing with another index, icky)
			for(     j = 0 ; j < sub_lim_y ; j++){  // j is row
				memcpy( &sub_a[ j*blockDim.x ], &a[ (sub_a_row+j)*len + sub_a_col ],  sub_lim_x*sizeof(float)); // copy row
			}
			// load b row by row
			for(     j = 0 ; j <  sub_lim_y ; j++){    // j is row
				memcpy( &sub_b[ j*blockDim.x ], &b[ (sub_b_row+j)*len + sub_b_col ],  sub_lim_x*sizeof(float)); // copy row
			}
		}
		// sync, other threads need to wait for data
		__syncthreads();
		
		// scan the submatrix, computing partial sum
		for( j=0 ; j < sub_lim_x ; j++ ){ 
			sum +=  sub_a[ threadIdx.y*blockDim.x + j ] * sub_b[ j*blockDim.x + threadIdx.x ];
		}
	
		// sync threads again before moving on to next tile
		__syncthreads();

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
/*	NUM_THREADS.x = NUM_THREADS.y = 16;
	blks.x = blks.y = (len_a + NUM_THREADS.x - 1 ) / NUM_THREADS.x;
	NUM_THREADS.z = blks.z = 1;
	printf("NUM_THREADS(%4u,%4u,   0)\n       blks(%4u,%4u,   0)\n",NUM_THREADS.x,NUM_THREADS.y,blks.x,blks.y);
	// launch kernel to do a*b=c
	matmul_kernel <<< blks, NUM_THREADS >>> (len_a , d_a , d_b , d_c);
*/	// launch kernel for shared memory implementation
	NUM_THREADS.x   = NUM_THREADS.y = 16;
	blks.x = blks.y = (len_a + NUM_THREADS.x - 1 ) / NUM_THREADS.x;
	NUM_THREADS.z   = blks.z = 1;
	unsigned shared_mem_size = 2*NUM_THREADS.y*NUM_THREADS.x*sizeof(float);
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