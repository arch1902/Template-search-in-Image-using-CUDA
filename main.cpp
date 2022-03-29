#include <iostream>
#include <string>
#include <fstream>

using namespace std;

// __global__ void add(int *a, int *b, int *c)
// {
//   *c = *a + *b;
// }

void input(int n, int m, string filename, int *arr, bool flag, int* avg)
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
  file.close();
  if(flag)*avg=val;
}


// TODO optimize this function, add elements row-wise in parallel and then take their sum
// User thread shared memory
__global__
void computeImageSummary(int *data, int n, int m, int query_n, int query_m, int *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx/m;
    int y = idx%m;
    long long val = 0;
    
    // printf("Inside, 1.)%d\t2.)%d\t3.)%d\t4.)%d\t5.)Idx:%d\n",x,y,query_n,query_m,idx);
    // printf("N,M, 1.)%d\t2.)%d\n",n,m);
    // if(x+query_n>=n || y+query_m>=m)return;
    

    for(int i=0;i<query_n;i++){
        for(int j=0;j<query_m;j++){
            val += data[(x+i)*m+(y+j)];
        }
    }
    result[idx] = (int)(val)/(query_n*query_m);
    // printf("Sub region (%d,%d) avg value: %d\n",x,y,idx,result[idx]);

    // print blockDim
    printf("BlockDim: %d\t, BlockIdx: %d\t, ThreadIdx: %d\n",blockDim.x,blockIdx.x,threadIdx.x);
}

int main(int argc, char* argv[]){

    string data_image_path = argv[1];
    string query_image_path = argv[2];
    double threshold = stod(argv[3]);
    int n = stoi(argv[4]);
    int rows, cols;
    int query_rows, query_cols;
    int imageSummaryQuery;

    // Read the data image
    ifstream data_image_file(data_image_path);
    data_image_file >> rows >> cols;
    data_image_file.close();
    int data_image[rows][cols];

    input(rows,cols,data_image_path,&data_image[0][0],false,&imageSummaryQuery);

    // Read the query image
    ifstream query_image_file(query_image_path);
    string query_image_line;
    query_image_file >> query_rows >> query_cols;
    query_image_file.close();
    int query_image[query_rows][query_cols];
    
    input(query_rows,query_cols,query_image_path,&query_image[0][0],true,&imageSummaryQuery);

    // Compute the image summary
    // int imageSummaryCuda[rows-query_rows][cols-query_cols];

    int block_size = 256;
    int grid_size = (rows-query_rows)/block_size + 1;

    int imageSummarySize = (cols-query_cols)*(rows-query_rows);

    int *data_imageCuda, *imageSummaryCuda;
    cudaMalloc((void **)&data_imageCuda,sizeof(int)*rows*cols);
    cudaMalloc((void **)&imageSummaryCuda,sizeof(int)*imageSummarySize);
    
    cudaMemcpy(data_imageCuda,&data_image[0][0],sizeof(int)*rows*cols,cudaMemcpyHostToDevice);

    // computeImageSummary<<<grid_size,block_size>>>(data_imageCuda,rows,cols,query_rows,query_cols,imageSummaryCuda);
    computeImageSummary<<<rows-query_rows,cols-query_cols>>>(data_imageCuda,rows,cols,query_rows,query_cols,imageSummaryCuda);
    // imageSummaryCuda

    cudaDeviceSynchronize();

    return 0;

    // for(int i=0;i<rows-query_rows;i++)
    // {
    //   for(int j=0;j<cols-query_cols;j++)
    //     cout << imageSummary[i][j] << ", ";cout << "\n";
    // }


    // int a, b, c; // host copies of a, b, c
    // int *d_a, *d_b, *d_c; // device copies of a, b, c
    // int size = sizeof(int);
    
    // // Allocate space for device copies of a, b, c
    // cudaMalloc((void **)&d_a, size);
    // cudaMalloc((void **)&d_b, size);
    // cudaMalloc((void **)&d_c, size);
    
    // // Setup input values
    // a = 2;
    // b = 7;
    
    // // Copy inputs to device
    // cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    
    // // Launch add() kernel on GPU
    // add<<<1,1>>>(d_a, d_b, d_c);
    
    // // Copy result back to host
    // cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    // cout << "Val: " << c << "\n";
    
    // // Cleanup
    // cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    // return 0;

}