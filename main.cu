#include <iostream>
#include <string>
#include <fstream>

using namespace std;

// __global__ void add(int *a, int *b, int *c)
// {
//   *c = *a + *b;
// }

void input(int n, int m, string filename, float *arr, int *R, int *G, int *B, bool flag, int &avg)
{
  int r,g,b;
  int val;
  ifstream file(filename);
  for(int i=0;i<n;i++){
      for(int j=0;j<m;j++){
          file >> r >> g >> b;
          R[i*m + j] = r;
          G[i*m + j] = g;
          B[i*m + j] = b;

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


__device__
void computeRMSD(float *dataR, float *dataG, float *dataB, int *queryR, int *queryG, int *queryB, int n, int m, int query_n, int query_m, int x, int y, int &rmsd)
{
    float sum = 0;
    for(int i=0;i<query_n;i++){
        for(int j=0;j<query_m;j++){
        sum += pow((dataR[(x+i)*m + (y+j)] - queryR[i*query_m + j]),2);
        sum += pow((dataG[(x+i)*m + (y+j)] - queryG[i*query_m + j]),2);
        sum += pow((dataB[(x+i)*m + (y+j)] - queryB[i*query_m + j]),2);
        }
    }
    rmsd = sqrt(sum);
}

// TODO store R,G,B pointers in some array or in some sort of struct
__global__
void computeImageSummary(float *data, float *dataR, float *dataG, float *dataB, int *queryData, int *queryR, int *queryG, int *queryB, int n, int m, int query_n, int query_m, float *result, int QueryVal, int threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx> m*n*3) return;
    int x = idx/(m*3);
    int y = (idx%(m*3))/3;
    int orientation = (idx%(m*3))%3;
    long long val = 0;
    //printf("x %d y %d or %d \n",x,y,orientation);
    
    // printf("Inside, 1.)%d\t2.)%d\t3.)%d\t4.)%d\t5.)Idx:%d\n",x,y,query_n,query_m,idx);
    // printf("N,M, 1.)%d\t2.)%d\n",n,m);
    // if(x+query_n>=n || y+query_m>=m)return;

    int xmin,xmax,ymin,ymax;

    if(orientation==0){
        xmin = 0;
        xmax = query_n;
        ymin = 0;
        ymax = query_m;
    }else if(orientation==1){ // +45 degrees
        ymin = -(query_n)/(1.414);
        ymax = query_m/(1.414);
        xmin = 0;
        xmax = (query_m + query_n)/(1.414);
    }else { // -45 degrees
        ymin = 0;
        ymax = (query_m + query_n)/(1.414);
        xmin = -(query_m)/(1.414);
        xmax = (query_n)/(1.414);
    }
    

    for(int i=xmin;i<xmax;i++){
        for(int j=ymin;j<ymax;j++){
            if(x+i >= n or x+i < 0 or y+j >= m or y+j < 0) val += 255;
            else val += data[(x+i)*m + (y+j)];
        }
    }

    int boxSize = (xmax-xmin)*(ymax-ymin);

    result[idx] = (float)(val)/boxSize;

    if(abs(result[idx] -QueryVal)<=threshold)
    {
        computeRMSD(dataR,dataG,dataB,queryR,queryG,queryB,n,m,query_n,query_m,x,y,rmsd);
        // computeRMSD(data,queryData,n,m,query_n,query_m,x,y);
    }

    // if(result[idx]>71.5 and result[idx]<72.5) {
    //     printf("%d %d %f \n",x,y,result[idx]);
    // }

    // if(x==290 and y==120 and orientation==1){
    //     printf("%.6f \n", result[idx]);
    // }

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
    float *data_imageV;
    int *data_imageR;
    int *data_imageG;
    int *data_imageB;
    cudaMallocManaged(&data_imageV, rows*cols*sizeof(float));
    cudaMallocManaged(&data_imageR, rows*cols*sizeof(int));
    cudaMallocManaged(&data_imageG, rows*cols*sizeof(int));
    cudaMallocManaged(&data_imageB, rows*cols*sizeof(int));

    input(rows,cols,data_image_path,data_imageV,data_imageR,data_imageG,data_imageB,false,imageSummaryQuery);

    // Read the query image
    ifstream query_image_file(query_image_path);
    int query_rows, query_cols;
    query_image_file >> query_rows >> query_cols;
    query_image_file.close();
    float *query_imageV;
    int *query_imageR;
    int *query_imageG;
    int *query_imageB;

    cudaMallocManaged(&query_imageV, query_rows*query_cols*sizeof(float));
    cudaMallocManaged(&query_imageR, query_rows*query_cols*sizeof(int));
    cudaMallocManaged(&query_imageG, query_rows*query_cols*sizeof(int));
    cudaMallocManaged(&query_imageB, query_rows*query_cols*sizeof(int));
    
    input(query_rows,query_cols,query_image_path,query_imageV,query_imageR,query_imageG,query_imageB,true,imageSummaryQuery);

    cout<<"Query Image Summary: "<<imageSummaryQuery<<endl;

    // Compute the image summary
    // int imageSummaryCuda[rows-query_rows][cols-query_cols];

    // int block_size = 256;
    // int grid_size = (rows-query_rows)/block_size + 1;

    int imageSummarySize = (cols)*(rows)*3;

    float *imageSummary;
    // cudaMalloc((void **)&data_imageVCuda,sizeof(int)*rows*cols);
    // cudaMalloc((void **)&imageSummaryCuda,sizeof(int)*imageSummarySize);
    
    // cudaMemcpy(data_imageVCuda,&data_imageV,sizeof(int)*rows*cols,cudaMemcpyHostToDevice);


    cudaMallocManaged(&imageSummary, imageSummarySize*sizeof(float));

    int num_blocks = (rows*cols*3)/1024 + 1;

    // computeImageSummary<<<grid_size,block_size>>>(data_imageVCuda,rows,cols,query_rows,query_cols,imageSummaryCuda);

    computeImageSummary<<<num_blocks, 1024>>>(data_imageV,query_imageV,rows,cols,query_rows,query_cols,imageSummary,imageSummaryQuery,threshold1);

    cudaError_t err = cudaGetLastError();

     if ( err != cudaSuccess )
     {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       

        // Possibly: exit(-1) if program cannot continue....
     }


    cudaDeviceSynchronize();

    // for(int i=0;i<imageSummarySize;i++){
    //     cout<<"x:"<<i/(cols*3)<<", y:"<<(i%(cols*3))/3<<", orientation:"<<(i%(cols*3))%3<<" -> "<<imageSummary[i]<<endl;
    // }


    return 0;

}