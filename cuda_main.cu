#include <iostream>
#include <string>
#include <fstream>

using namespace std;

// __global__ void add(int *a, int *b, int *c)
// {
//   *c = *a + *b;
// }

void input(int n, int m, ifstream file, int *arr, bool flag, int* avg)
{
  int r,g,b;
  int val;
  for(int i=0;i<n;i++){
      for(int j=0;j<m;j++){
          file >> r >> g >> b;
          //TODO check if conversion to float array is required or not
          arr[i*m + j] = (r+g+b)/3;
          val += arr[i*m+j];
      }
  }
  if(flag)*avg=val;
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
    int data_image[rows][cols];
    input(rows,cols,data_image_file,data_image,false,&imageSummaryQuery);
    data_image_file.close();

    // Read the query image
    ifstream query_image_file(query_image_path);
    string query_image_line;
    query_image_file >> query_rows >> query_cols;
    int query_image[query_rows][query_cols];
    input(query_rows,query_cols,query_image_file,query_image,true,&imageSummaryQuery);
    query_image_file.close();




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