#include <iostream>
#include <string>
#include <fstream>

using namespace std;

// __global__ void add(int *a, int *b, int *c)
// {
//   *c = *a + *b;
// }

void input(int n, int m, string filename, float *arr, bool flag, int &avg)
{
  int r,g,b;
  int val;
  ifstream file(filename);
  for(int i=0;i<n;i++){
      for(int j=0;j<m;j++){
          file >> r >> g >> b;
          //TODO check if conversion to float array is required or not
          arr[i*m + j] = (r+g+b)/3;
          val += arr[i*m+j];
      }
  }
  if(!flag) cout<<"Data Image read !"<<endl;
  file.close();
  if(flag) avg = val/(m*n);
}


// TODO optimize this function, add elements row-wise in parallel and then take their sum
// User thread shared memory
__global__
void computeImageSummary(float *data, int n, int m, int query_n, int query_m, float *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = ((idx - threadIdx.x)/3)/(m-query_m);
    int y = ((idx - threadIdx.x)/3)%(m-query_m);
    int orientation = threadIdx.x;
    long long val = 0;
    printf("x %d y %d or %d \n",x,y,orientation);
    
    // printf("Inside, 1.)%d\t2.)%d\t3.)%d\t4.)%d\t5.)Idx:%d\n",x,y,query_n,query_m,idx);
    // printf("N,M, 1.)%d\t2.)%d\n",n,m);
    // if(x+query_n>=n || y+query_m>=m)return;
    

    for(int i=0;i<query_n;i++){
        for(int j=0;j<query_m;j++){
            val += data[(x+i)*m+(y+j)];
        }
    }

    result[idx] = (float)(val)/(query_n*query_m);
    // printf("Sub region (%d,%d) avg value: %d\n",x,y,idx,result[idx]);

    // print blockDim
    //printf("BlockDim: %d\t, BlockIdx: %d\t, ThreadIdx: %d\n",blockDim.x,blockIdx.x,threadIdx.x);
}

int main(int argc, char* argv[]){

    string data_image_path = argv[1];
    string query_image_path = argv[2];
    double threshold1 = stod(argv[3]);
    double threshold2 = stod(argv[4]);
    int n = stoi(argv[5]);
    int rows, cols;
    int imageSummaryQuery;

    // Read the data image
    ifstream data_image_file(data_image_path);
    data_image_file >> rows >> cols;
    data_image_file.close();
    float *data_image;
    cudaMallocManaged(&data_image, rows*cols*sizeof(float));

    input(rows,cols,data_image_path,data_image,false,imageSummaryQuery);

    // Read the query image
    ifstream query_image_file(query_image_path);
    int query_rows, query_cols;
    query_image_file >> query_rows >> query_cols;
    query_image_file.close();
    float query_image[query_rows*query_cols];
    
    input(query_rows,query_cols,query_image_path,query_image,true,imageSummaryQuery);

    cout<<"Query Image Summary: "<<imageSummaryQuery<<endl;

    // Compute the image summary
    // int imageSummaryCuda[rows-query_rows][cols-query_cols];

    // int block_size = 256;
    // int grid_size = (rows-query_rows)/block_size + 1;

    int imageSummarySize = (cols-query_cols)*(rows-query_rows)*3;

    float *imageSummary;
    // cudaMalloc((void **)&data_imageCuda,sizeof(int)*rows*cols);
    // cudaMalloc((void **)&imageSummaryCuda,sizeof(int)*imageSummarySize);
    
    // cudaMemcpy(data_imageCuda,&data_image,sizeof(int)*rows*cols,cudaMemcpyHostToDevice);


    cudaMallocManaged(&imageSummary, imageSummarySize*sizeof(float));

    int num_blocks = (rows-query_rows)*(cols-query_cols);

    // computeImageSummary<<<grid_size,block_size>>>(data_imageCuda,rows,cols,query_rows,query_cols,imageSummaryCuda);

    computeImageSummary<<<num_blocks, 3>>>(data_image,rows,cols,query_rows,query_cols,imageSummary);

    cudaError_t err = cudaGetLastError();

     if ( err != cudaSuccess )
     {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       

        // Possibly: exit(-1) if program cannot continue....
     }


    cudaDeviceSynchronize();

    // for(int i=0;i<imageSummarySize;i++){
    //     cout<<"x:"<<i/((cols-query_cols)*3)<<", y:"<<(i%((cols-query_cols)*3))/3<<", orientation:"<<(i%((cols-query_cols)*3))%3<<" -> "<<imageSummary[i]<<endl;
    // }


    return 0;

}