#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;

void dataInput(int n, int m, string filename, float *arr, int *R, int *G, int *B, int *compressedData)
{
  int r,g,b;
  float val = 0;
  ifstream file(filename);
  file >> n >> m;

  for(int i=n-1;i>=0;i--)
  {
      float valRow = 0;
      for(int j=0;j<m;j++){
          file >> r >> g >> b;
          R[i*m + j] = r;
          G[i*m + j] = g;
          B[i*m + j] = b;

          //TODO check if conversion to float array is required or not
          arr[i*m + j] = (r+g+b)/3;
          valRow += arr[i*m+j];
          compressedData[i*m + j] = valRow;
      }
      val += valRow;
  }
  cout<<"Data Image read !"<<endl;
  file.close();
}

void queryInput(int n, int m, string filename, float *arr, int *R, int *G, int *B, float &avg)
{
  int r,g,b;
  float val = 0;
  ifstream file(filename);
  file >> n >> m;

  for(int i=n-1;i>=0;i--){
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
  cout << "Query Image read!\n";
  file.close();
  avg = val/(m*n);
}

// TODO optimize this function, add elements row-wise in parallel and then take their sum
// User thread shared memory

__device__
void bilinearInterpolation(int *data, float x, float y,int n, int m, float &val)
{
    float upx    = ceil(x);
    float downx  = floor(x);
    float righty = ceil(y);
    float lefty  = floor(y);

    float topleft;
    float topright;
    float bottomleft;
    float bottomright;

    if(upx>=n or lefty<0){
        topleft = 0;
    }else {
        topleft = data[int(upx)*m+int(lefty)];
    }

    if(upx>=n or righty>=m){
        topright = 0;
    }else {
        topright = data[int(upx)*m+int(righty)];
    }

    if(downx<0 or lefty<0){
        bottomleft = 0;
    }else {
        bottomleft = data[int(downx)*m+int(lefty)];
    }

    if(downx<0 or righty>=m){
        bottomright = 0;
    }else {
        bottomright = data[int(downx)*m+int(righty)];
    }   

    // F(x,y) = z00*(1-x)*(1-y) + z10*x*(1-y) + z01*(1-x)*y + z11*x*y
    float f = bottomleft*(righty - y)*(upx - x) + topleft*(righty - y)*(x - downx) + bottomright*(y-lefty)*(upx - x) + topright*(y-lefty)*(x-downx);
    val = f;
}

__device__
void computeRMSD(int *dataR, int *dataG, int *dataB, int *queryR, int *queryG, int *queryB, int n, int m, int query_n, int query_m, int idx, float &rmsd)
{
    int x = idx/(m*3);
    int y = (idx%(m*3))/3;
    // (x,y) is the bottom left pixel coordinate of the data image
    int orientation = (idx%(m*3))%3;
    float sum  = 0;
    // float temp = 0;

    float ptx,pty;

    for(int i=0;i<query_n;i++)
    {
        for(int j=0;j<query_m;j++)
        {
            int query_cord = int(i)*query_m+int(j);
            if(orientation==0)
            {
                ptx = x+i;
                pty = y+j;

                if(ptx <0 or ptx>=n or pty <0 or pty>=m){
                    sum += (queryR[query_cord])*(queryR[query_cord]);
                    sum += (queryG[query_cord])*(queryG[query_cord]);
                    sum += (queryB[query_cord])*(queryB[query_cord]);
                    continue;
                }

                int data_cord = int(ptx)*m+int(pty);
                
                sum += (dataR[data_cord] - queryR[query_cord])*(dataR[data_cord] - queryR[query_cord]);
                sum += (dataG[data_cord] - queryG[query_cord])*(dataG[data_cord] - queryG[query_cord]);
                sum += (dataB[data_cord] - queryB[query_cord])*(dataB[data_cord] - queryB[query_cord]);
            }
            else if(orientation==1) // +45
            {
                ptx = x + i*(1/sqrt(2.0)) + j*(1/sqrt(2.0));
                pty = y + j*(1/sqrt(2.0)) - i*(1/sqrt(2.0));

                float r;bilinearInterpolation(dataR,ptx,pty,n,m,r);
                float g;bilinearInterpolation(dataG,ptx,pty,n,m,g);
                float b;bilinearInterpolation(dataB,ptx,pty,n,m,b);

                sum += (r - queryR[query_cord])*(r - queryR[query_cord]);
                sum += (g - queryG[query_cord])*(g - queryG[query_cord]);
                sum += (b - queryB[query_cord])*(b - queryB[query_cord]);

            }
            else if(orientation==2)
            {
                ptx = x + i*(1/sqrt(2.0)) - j*(1/sqrt(2.0));
                ptx = x + i*(1/sqrt(2.0)) + j*(1/sqrt(2.0));

                float r;bilinearInterpolation(dataR,ptx,pty,n,m,r);
                float g;bilinearInterpolation(dataG,ptx,pty,n,m,g);
                float b;bilinearInterpolation(dataB,ptx,pty,n,m,b);

                sum += (r - queryR[query_cord])*(r - queryR[query_cord]);
                sum += (g - queryG[query_cord])*(g - queryG[query_cord]);
                sum += (b - queryB[query_cord])*(b - queryB[query_cord]);
            }
        }
    }

    rmsd = sqrt(sum/(query_n*query_m*3));

}

// TODO store R,G,B pointers in some array or in some sort of struct
__global__
void computeImageSummary(float *data, int *dataR, int *dataG, int *compressedData, int *dataB, float *queryData, int *queryR, int *queryG, int *queryB, int n, int m, int query_n, int query_m, float *result, float *rmsdValues, int QueryVal, float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx> m*n*3) return;
    int x = idx/(m*3);
    int y = (idx%(m*3))/3;
    int orientation = (idx%(m*3))%3;
    long long val = 0;
    //printf("x %d y %d or %d \n",x,y,orientation);

    int xmin,xmax,ymin,ymax;

    if(orientation==0){
        xmin = 0;
        xmax = query_n;
        ymin = 0;
        ymax = query_m;
    }else if(orientation==1) { // +45 degrees
        ymin = -(query_n)/(sqrt(2.0));
        ymax = query_m/(sqrt(2.0));
        xmin = 0;
        xmax = (query_m + query_n)/(sqrt(2.0));
    }else { // -45 degrees
        ymin = 0;
        ymax = (query_m + query_n)/(sqrt(2.0));
        xmin = -(query_m)/(sqrt(2.0));
        xmax = (query_n)/(sqrt(2.0));
    }
    
    for(int i=x+xmin;i<x+xmax;i++){
        for(int j=y+ymin;j<y+ymax;j++){
            if(x+i >= n or x+i < 0 or y+j >= m or y+j < 0) val += 0;
            else val += data[(x+i)*m + (y+j)];
        }
    }

    if((x+xmin)<0 || (x+xmax)>=n || (y+ymin)<0 || (y+ymax)>=m)
    {
        result[idx] = -1;
        rmsdValues[idx] = -1;
        return;
    }

    for(int i=x+xmin;i<x+xmax;i++)
    {
        val += (compressedData[i*m + (y+ymax)] - compressedData[i*m + (y+ymin)] + data[i*m + y+ymin]);
    }

    // for(int i=xmin;i<xmax;i++){
    //     for(int j=ymin;j<ymax;j++){
    //         if(x+i >= n or x+i < 0 or y+j >= m or y+j < 0) val += 0;
    //         else val += data[(x+i)*m + (y+j)];
    //     }
    // }

    int boxSize = (xmax-xmin)*(ymax-ymin);

    result[idx] = (float)(val)/boxSize;

    float rmsd = -1;

    if(abs(result[idx] - QueryVal) <= threshold)
    {
        computeRMSD(dataR,dataG,dataB,queryR,queryG,queryB,n,m,query_n,query_m,idx,rmsd);
    }

    rmsdValues[idx] = rmsd;

    // if(result[idx]>71.5 and result[idx]<72.5) {
    //     printf("%d %d %f \n",x,y,result[idx]);
    // }

    // if(x==290 and y==120 and orientation==1){
    //     printf("RMSD:%.6f AVG:%f %f %d \n", rmsdValues[idx], result[idx], threshold, QueryVal);
    // }
}

struct triplet
{
    int x,y;
    int orientation; // 0 for 0 degrees, 1 for 45 degrees, 2 for -45 degrees
    float val;
};

bool sortbyVal(const triplet &a, 
              const triplet &b) 
{ 
    return (a.val < b.val);
}

int main(int argc, char* argv[]){

    string data_image_path = argv[1];
    string query_image_path = argv[2];
    float threshold1 = stof(argv[3]); // for RMSD
    float threshold2 = stof(argv[4]); // for Gray-Scale image summary
    int n = stoi(argv[5]);
    int rows, cols;
    float imageSummaryQuery;

    // Read the data image
    ifstream data_image_file(data_image_path);
    data_image_file >> rows >> cols;
    data_image_file.close();
    float *data_imageV;
    int *data_imageR;
    int *data_imageG;
    int *data_imageB;
    int *compressedData;

    cudaMallocManaged(&data_imageV, rows*cols*sizeof(float));
    cudaMallocManaged(&data_imageR, rows*cols*sizeof(int));
    cudaMallocManaged(&data_imageG, rows*cols*sizeof(int));
    cudaMallocManaged(&data_imageB, rows*cols*sizeof(int));
    cudaMallocManaged(&compressedData, rows*cols*sizeof(int));

    dataInput(rows,cols,data_image_path,data_imageV,data_imageR,data_imageG,data_imageB,compressedData);

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
    
    queryInput(query_rows,query_cols,query_image_path,query_imageV,query_imageR,query_imageG,query_imageB,imageSummaryQuery);

    cout<<"Query Image Summary: "<<imageSummaryQuery<<endl;

    int imageSummarySize = (cols)*(rows)*3;

    float *imageSummary;
    float *rmsdValues;


    cudaMallocManaged(&imageSummary, imageSummarySize*sizeof(float));
    cudaMallocManaged(&rmsdValues, imageSummarySize*sizeof(float));

    int num_blocks = (rows*cols*3)/1024 + 1;

    computeImageSummary<<<num_blocks, 1024>>>(data_imageV,data_imageR, data_imageG, compressedData, data_imageB, query_imageV, query_imageR, query_imageG, query_imageB, rows, cols, query_rows, query_cols, imageSummary, rmsdValues, imageSummaryQuery,threshold2);

    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));       

    // Possibly: exit(-1) if program cannot continue....
    }

    cudaDeviceSynchronize();

    vector<triplet> output;

    for(int i=0;i<imageSummarySize;i++)
    {
        if(rmsdValues[i]!=-1)
        {
            triplet t;
            t.x   = i/(cols*3);
            t.y   = (i%(cols*3))/3;
            t.orientation = (i%(cols*3))%3;
            t.val = rmsdValues[i];
            output.push_back(t);
        }
    }
    sort(output.begin(),output.end(),sortbyVal);

    for(int i=0;i<min(n,(int)output.size());i++)
    {
        cout << "x:" << output[i].x << ", y:" << output[i].y << ", orientation:" << output[i].orientation << ", val:" << output[i].val << "\n";
    }
    cout<<output.size()<<endl;

    // for(int i=0;i<imageSummarySize;i++){
    //     cout<<"x:"<<i/(cols*3)<<", y:"<<(i%(cols*3))/3<<", orientation:"<<(i%(cols*3))%3<<" -> "<<imageSummary[i]<<endl;
    // }


    return 0;

}