include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include <cstdlib>

using namespace std;

void multiply_mv_row_major(const double *matrix, int rows, int cols,
                           const double *vector, double *result) {
  if (!matrix || !vector || !result) {
    cerr << "Error: Null pointer input to multiply_mv_row_major" << endl;
    return;
  }
  for (int i = 0; i < rows; i++) {
    result[i] = 0.0;
    for (int j = 0; j < cols; j++) {
      result[i] += matrix[i * cols + j] * vector[j];
    }
  }
}

void multiply_mv_col_major(const double *matrix, int rows, int cols,
                           const double *vector, double *result) {
  if (!matrix || !vector || !result) {
    cerr << "Error: Null pointer input to multiply_mv_col_major" << endl;
    return;
  }
  for (int i = 0; i < rows; i++) {
    result[i] = 0.0;
    for (int j = 0; j < cols; j++) {
      result[i] += matrix[j * rows + i] * vector[j];
    }
  }
}

void multiply_mm_naive(const double *matrixA, int rowsA, int colsA,
                       const double *matrixB, int rowsB, int colsB,
                       double *result) {
  if (!matrixA || !matrixB || !result) {
    cerr << "Error: Null pointer input to multiply_mm_naive" << endl;
    return;
  }
  if (colsA != rowsB) {
    cerr << "Error: Incompatible dimensions for matrix multiplication" << endl;
    return;
  }
  for (int i = 0; i < rowsA; ++i) {
    for (int j = 0; j < colsB; ++j) {
      result[i * colsB + j] = 0.0;
      for (int k = 0; k < colsA; ++k) {
        result[i * colsB + j] +=
            matrixA[i * colsA + k] * matrixB[k * colsB + j];
      }
    }
  }
}

void multiply_mm_transposed_b(const double *matrixA, int rowsA, int colsA,
                              const double *matrixB_transposed, int rowsB,
                              int colsB, double *result) {
  if (!matrixA || !matrixB_transposed || !result) {
    cerr << "Error: Null pointer input to multiply_mm_transposed_b" << endl;
    return;
  }
  if (colsA != rowsB) {
    cerr << "Error: Incompatible dimensions for matrix multiplication" << endl;
    return;
  }
  for (int i = 0; i < rowsA; ++i) {
    for (int j = 0; j < colsB; ++j) {
      result[i * colsB + j] = 0.0;
      for (int k = 0; k < colsA; ++k) {
        result[i * colsB + j] +=
            matrixA[i * colsA + k] * matrixB_transposed[j * rowsB + k];
      }
    }
  }
}

template <typename Func>
void benchmark(Func f, int N, double &mean, double &stddev) {
  vector<double> times(N);
  for (int i = 0; i < N; ++i) {
    auto start = chrono::high_resolution_clock::now();
    f();
    auto end = chrono::high_resolution_clock::now();
    times[i] = chrono::duration<double>(end - start).count();
  }
  mean = accumulate(times.begin(), times.end(), 0.0) / N;
  double variance = accumulate(times.begin(), times.end(), 0.0,
                               [mean](double acc, double t) {
                                 return acc + (t - mean) * (t - mean);
                               }) /
                    N;
  stddev = sqrt(variance);
}

int main() {
  // Testing Function 1 & 2:
  int rows = 2, cols = 3;
  double *row_major_matrix = new double[6]{1, 2, 3, 4, 5, 6};
  double *col_major_matrix = new double[6]{1, 4, 2, 5, 3, 6};
  double *vec = new double[3]{7, 8, 9};

  double *result_row = new double[2];
  double *result_col = new double[2];

  multiply_mv_row_major(row_major_matrix, rows, cols, vec, result_row);
  multiply_mv_col_major(col_major_matrix, rows, cols, vec, result_col);

  cout << "Row major result: " << result_row[0] << " " << result_row[1] << endl;
  cout << "Col major result: " << result_col[0] << " " << result_col[1] << endl;
  if (result_row[0] == 50 && result_row[1]) {
    cout << "Row major result correct" << endl;
  } else {
    cout << "Row major needs some help" << endl;
  }

  if (result_col[0] == 50 && result_col[1]) {
    cout << "Col major result correct" << endl;
  } else {
    cout << "Col major needs some help" << endl;
  }
  // expected: 50 122

  delete[] row_major_matrix;
  delete[] col_major_matrix;
  delete[] vec;
  delete[] result_row;
  delete[] result_col;

  // Testing for function 3 and 4:
  // A is 2x3, B is 3x2, result is 2x2
  // Expected: A * B = [[58, 64], [139, 154]]
  int rowsA = 2, colsA = 3, rowsB = 3, colsB = 2;

  double *matA = new double[6]{1, 2, 3, 4, 5, 6};
  double *matB = new double[6]{7, 8, 9, 10, 11, 12};
  // B transposed (2x3): rows and cols of B swapped
  double *matB_T = new double[6]{7, 9, 11, 8, 10, 12};

  double *result_naive = new double[4];
  double *result_transposed = new double[4];

  multiply_mm_naive(matA, rowsA, colsA, matB, rowsB, colsB, result_naive);
  multiply_mm_transposed_b(matA, rowsA, colsA, matB_T, rowsB, colsB,
                           result_transposed);

  cout << "Naive result: " << result_naive[0] << " " << result_naive[1] << " "
       << result_naive[2] << " " << result_naive[3] << endl;
  cout << "Transposed B result: " << result_transposed[0] << " "
       << result_transposed[1] << " " << result_transposed[2] << " "
       << result_transposed[3] << endl;

  if (result_naive[0] == 58 && result_naive[1] == 64 &&
      result_naive[2] == 139 && result_naive[3] == 154)
    cout << "Naive result correct" << endl;
  else
    cout << "Naive result incorrect" << endl;

  if (result_transposed[0] == 58 && result_transposed[1] == 64 &&
      result_transposed[2] == 139 && result_transposed[3] == 154)
    cout << "Transposed B result correct" << endl;
  else
    cout << "Transposed B result incorrect" << endl;

  delete[] matA;
  delete[] matB;
  delete[] matB_T;
  delete[] result_naive;
  delete[] result_transposed;

  // Benchmarking
  cout << "\n--- Benchmarking ---\n";
  cout << "Size\t\tmv_row(us)\tmv_col(us)\tmm_naive(us)\tmm_transposed(us)\n";

  for (int n : {32, 128, 512, 1024}) {
    double mean, stddev;
    int runs = 20;

    // mv row major
    double *mvA_row = new double[n * n];
    double *mvVec_row = new double[n];
    double *mvRes_row = new double[n];
    for (int i = 0; i < n * n; i++)
      mvA_row[i] = 1.0;
    for (int i = 0; i < n; i++)
      mvVec_row[i] = 1.0;
    benchmark(
        [&]() { multiply_mv_row_major(mvA_row, n, n, mvVec_row, mvRes_row); },
        runs, mean, stddev);
    volatile double dummy = mvRes_row[0]; // prevent optimization
    (void)dummy; // silence unused variable warning
    cout << n << "x" << n << "\t\t" << mean * 1e6;
    delete[] mvA_row;
    delete[] mvVec_row;
    delete[] mvRes_row;

    // mv col major
    double *mvA_col = new double[n * n];
    double *mvVec_col = new double[n];
    double *mvRes_col = new double[n];
    for (int i = 0; i < n * n; i++)
      mvA_col[i] = 1.0;
    for (int i = 0; i < n; i++)
      mvVec_col[i] = 1.0;
    benchmark(
        [&]() { multiply_mv_col_major(mvA_col, n, n, mvVec_col, mvRes_col); },
        runs, mean, stddev);
    volatile double dummy2 = mvRes_col[0]; // prevent optimization
    (void)dummy2; // silence unused variable warning
    cout << "\t\t" << mean * 1e6;
    delete[] mvA_col;
    delete[] mvVec_col;
    delete[] mvRes_col;

    // mm naive
    double *mmA = new double[n * n];
    double *mmB = new double[n * n];
    double *mmRes = new double[n * n];
    for (int i = 0; i < n * n; i++)
      mmA[i] = mmB[i] = 1.0;
    benchmark([&]() { multiply_mm_naive(mmA, n, n, mmB, n, n, mmRes); }, runs,
              mean, stddev);
    volatile double dummy3 = mmRes[0]; // prevent optimization
    (void)dummy3; // silence unused variable warning
    cout << "\t\t" << mean * 1e6;
    delete[] mmA;
    delete[] mmB;
    delete[] mmRes;

    // mm transposed b
    double *mmA2 = new double[n * n];
    double *mmBT = new double[n * n];
    double *mmRes2 = new double[n * n];
    for (int i = 0; i < n * n; i++)
      mmA2[i] = mmBT[i] = 1.0;
    benchmark(
        [&]() { multiply_mm_transposed_b(mmA2, n, n, mmBT, n, n, mmRes2); },
        runs, mean, stddev);
    volatile double dummy4 = mmRes2[0]; // prevent optimization
    (void)dummy4; // silence unused variable warning
    cout << "\t\t" << mean * 1e6 << "\n";
    delete[] mmA2;
    delete[] mmBT;
    delete[] mmRes2;
  }

  // Benchmark with aligned memory
  cout << "\n--- Benchmarking with aligned memory ---\n";
  cout << "Size\t\tmv_row(us)\tmv_col(us)\tmm_naive(us)\tmm_transposed(us)\n";

  for (int n : {32, 128, 512, 1024}) {
    double mean, stddev;
    int runs = 20;

    // mv row major
    double *mvA_row = (double *)aligned_alloc(64, n * n * sizeof(double));
    double *mvVec_row = (double *)aligned_alloc(64, n * sizeof(double));
    double *mvRes_row = (double *)aligned_alloc(64, n * sizeof(double));
    for (int i = 0; i < n * n; i++)
      mvA_row[i] = 1.0;
    for (int i = 0; i < n; i++)
      mvVec_row[i] = 1.0;
    benchmark(
        [&]() { multiply_mv_row_major(mvA_row, n, n, mvVec_row, mvRes_row); },
        runs, mean, stddev);
    volatile double dummy = mvRes_row[0]; // prevent optimization
    (void)dummy; // silence unused variable warning
    cout << n << "x" << n << "\t\t" << mean * 1e6;
    free(mvA_row);
    free(mvVec_row);
    free(mvRes_row);

    // mv col major
    double *mvA_col = (double *)aligned_alloc(64, n * n * sizeof(double));
    double *mvVec_col = (double *)aligned_alloc(64, n * sizeof(double));
    double *mvRes_col = (double *)aligned_alloc(64, n * sizeof(double));
    for (int i = 0; i < n * n; i++)
      mvA_col[i] = 1.0;
    for (int i = 0; i < n; i++)
      mvVec_col[i] = 1.0;
    benchmark(
        [&]() { multiply_mv_col_major(mvA_col, n, n, mvVec_col, mvRes_col); },
        runs, mean, stddev);
    volatile double dummy2 = mvRes_col[0]; // prevent optimization
    (void)dummy2; // silence unused variable warning
    cout << "\t\t" << mean * 1e6;
    free(mvA_col);
    free(mvVec_col);
    free(mvRes_col);

    // mm naive
    double *mmA = (double *)aligned_alloc(64, n * n * sizeof(double));
    double *mmB = (double *)aligned_alloc(64, n * n * sizeof(double));
    double *mmRes = (double *)aligned_alloc(64, n * n * sizeof(double));
    for (int i = 0; i < n * n; i++)
      mmA[i] = mmB[i] = 1.0;
    benchmark([&]() { multiply_mm_naive(mmA, n, n, mmB, n, n, mmRes); }, runs,
              mean, stddev);
    volatile double dummy3 = mmRes[0]; // prevent optimization
    (void)dummy3; // silence unused variable warning
    cout << "\t\t" << mean * 1e6;
    free(mmA);
    free(mmB);
    free(mmRes);

    // mm transposed b
    double *mmA2 = (double *)aligned_alloc(64, n * n * sizeof(double));
    double *mmBT = (double *)aligned_alloc(64, n * n * sizeof(double));
    double *mmRes2 = (double *)aligned_alloc(64, n * n * sizeof(double));
    for (int i = 0; i < n * n; i++)
      mmA2[i] = mmBT[i] = 1.0;
    benchmark(
        [&]() { multiply_mm_transposed_b(mmA2, n, n, mmBT, n, n, mmRes2); },
        runs, mean, stddev);
    volatile double dummy4 = mmRes2[0]; // prevent optimization
    (void)dummy4; // silence unused variable warning
    cout << "\t\t" << mean * 1e6 << "\n";
    free(mmA2);
    free(mmBT);
    free(mmRes2);
  }

  return 0;
}