#include <string>
#include <unordered_map>
#include <iostream>
#include "gtest/gtest.h"
#include "ortools/forecaster/fourier_forecaster.h"


namespace {
 using namespace operations_research::forecaster;

// The fixture for testing class Foo.
class FourierForecasterTest : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  FourierForecasterTest() {
     // You can do set-up work for each test here.
  }

  ~FourierForecasterTest() override {
     // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
     round_off_error_small_dataset_.insert({FourierForecaster::GLOP, DATA_REAL_VAL_TYPE(5e-16)});
     round_off_error_large_dataset_.insert({FourierForecaster::GLOP, DATA_REAL_VAL_TYPE(5e-14)});
     prepare_datasets();
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }

  void prepare_dataset_1() {

     // dataset 1 - simple test example with 4 points
     // x = [ 1, 1, 0, 0 ]
     //
     dataset1_len_ = 4;
     
     dataset1_.push_back({0,1.0});
     dataset1_.push_back({1,1.0});
     dataset1_.push_back({2,0.0});
     dataset1_.push_back({3,0.0});
  }

  void prepare_dataset_2() {

     // dataset 2
     // x = [ 2, 2, 0, 0 ]
     //
     dataset2_len_ = 4;
     
     dataset2_.push_back({0, 2.0});
     dataset2_.push_back({1, 2.0});
     dataset2_.push_back({2, 0.0});
     dataset2_.push_back({3, 0.0});
  }


  void prepare_dataset_3() {

     // dataset 3
     // x = [ 1, 0, 1, 0 ]
     //
     dataset3_len_ = 4;
     
     dataset3_.push_back({0, 1.0});
     dataset3_.push_back({1, 0.0});
     dataset3_.push_back({2, 1.0});
     dataset3_.push_back({3, 0.0});
  }

  void prepare_dataset_4() {

     // dataset 4 - ones on 2nd and on 100th position, everything else zero
     //
     dataset4_len_ = 500;
     for (int i = 0; i < dataset4_len_; ++i)
	     dataset4_.push_back({i,0.0});
     dataset4_[0].second=1.0;
     dataset4_[99].second=1.0;
  }

  std::unique_ptr<SparseDataContainer<DATA_REAL_VAL_TYPE>> prepare_dataset_5() {
     
     Inverse1DTransform ifftTr;
     auto& dataset5 = ifftTr.get_container(dataset5_len_);
     
     // dataset 5 - 5 non-zero frequencies + DC component, zeros everywhere else
     //
     dataset5[0][0]=1.0;
     dataset5[0][1]=0.0;
     dataset5[50][0]=0.0;
     dataset5[50][1]=0.0;

     dataset5[1][0] = 10;
     dataset5[1][1] = 10;

     dataset5[2][0] = 5;
     dataset5[2][1] = 1;

     dataset5[7][0] = 3;
     dataset5[7][1] = -2;

     dataset5[15][0] = 9;
     dataset5[15][1] = -1;

     dataset5[20][0] = 12;
     dataset5[20][1] = -0.5;

     dataset5[99][0] = 10;
     dataset5[99][1] = -10;

     dataset5[98][0] = 5;
     dataset5[98][1] = -1;

     dataset5[93][0] = 3;
     dataset5[93][1] = 2;

     dataset5[85][0] = 9;
     dataset5[85][1] = 1;

     dataset5[80][0] = 12;
     dataset5[80][1] = 0.5;

     LOG(INFO) << "Original Data: sumOfAbsYRe = " << dataset5.l1_norm()[0] << ", sumOfAbsYIm = " << dataset5.l1_norm()[1];

     ifftTr.execute();
     auto& time_signal_from_sparse_spectrum = ifftTr.get_result();
     const DATA_IDX_TYPE sampled_time_signal_len = dataset5_len_ / dataset5_sample_period_;
     SparseDataContainer<DATA_REAL_VAL_TYPE>* sampled_time_signal_from_sparse_spectrum = new SparseDataContainer<DATA_REAL_VAL_TYPE>(); 
     for (int i = 0; i < dataset5_len_; i+=dataset5_sample_period_) {
	(*sampled_time_signal_from_sparse_spectrum).push_back({i,time_signal_from_sparse_spectrum[i][0]});
     }
     return std::unique_ptr<SparseDataContainer<DATA_REAL_VAL_TYPE>>(sampled_time_signal_from_sparse_spectrum);
  }

  std::unique_ptr<SparseDataContainer<DATA_REAL_VAL_TYPE>> prepare_dataset_6() {
     // this is the result of dataset5
     Inverse1DTransform ifftTr;
     auto& dataset6 = ifftTr.get_container(dataset6_len_);
    
     // varX[0]=19.6626, varY[0]=0
     // varX[9]=0, varY[9]=3.18509
     // varX[13]=9.8458, varY[13]=0
     // varX[22]=4.43578, varY[22]=0
     // varX[23]=0.692778, varY[23]=0
     // varX[32]=0.197511, varY[32]=0
     // varX[33]=4.16919, varY[33]=0
     // varX[36]=0, varY[36]=-0.0716289
     // varX[48]=6.65334, varY[48]=0
     // varX[49]=3.67432, varY[49]=0
     // varX[50]=0, varY[50]=0
     // varX[51]=3.67432, varY[51]=0
     // varX[52]=6.65334, varY[52]=0
     // varX[64]=0, varY[64]=0.0716289
     // varX[67]=4.16919, varY[67]=0
     // varX[68]=0.197511, varY[68]=0
     // varX[77]=0.692778, varY[77]=0
     // varX[78]=4.43578, varY[78]=0
     // varX[87]=9.8458, varY[87]=0
     // varX[91]=0, varY[91]=-3.18509


     // dataset 6 - 9 non-zero frequencies + DC component, zeros everywhere else
     // there are 3 groups of 2 frequencies each 
     //
     dataset6[0][0]=19.6626;
     dataset6[0][1]=0.0;
     dataset6[50][0]=0.0;
     dataset6[50][1]=0.0;

     dataset6[9][0] = 0;
     dataset6[9][1] = 3.18509;

     dataset6[13][0] = 9.8458;
     dataset6[13][1] = 0;

     dataset6[22][0] = 4.43578;
     dataset6[22][1] = 0;

     dataset6[23][0] = 0.692778;
     dataset6[23][1] = 0;

     dataset6[32][0] = 0.197511;
     dataset6[32][1] = 0;

     dataset6[33][0] = 4.16919;
     dataset6[33][1] = 0;

     dataset6[36][0] = 0;
     dataset6[36][1] = -0.0716289;

     dataset6[48][0] = 6.65334;
     dataset6[48][1] = 0;

     dataset6[49][0] = 3.67432;
     dataset6[49][1] = 0;

     dataset6[51][0] = 3.67432;
     dataset6[51][1] = 0;

     dataset6[52][0] = 6.65334;
     dataset6[52][1] = 0;

     dataset6[64][0] = 0;
     dataset6[64][1] = 0.0716289;

     dataset6[67][0] = 4.16919;
     dataset6[67][1] = 0;

     dataset6[68][0] = 0.197511;
     dataset6[68][1] = 0;

     dataset6[77][0] = 0.692778;
     dataset6[77][1] = 0;

     dataset6[78][0] = 4.43578;
     dataset6[78][1] = 0;

     dataset6[87][0] = 9.8458;
     dataset6[87][1] = 0;

     dataset6[91][0] = 0;
     dataset6[91][1] = -3.18509;


     LOG(INFO) << "Original Data: sumOfAbsYRe = " << dataset6.l1_norm()[0] << ", sumOfAbsYIm = " << dataset6.l1_norm()[1];

     ifftTr.execute();
     auto& time_signal_from_sparse_spectrum = ifftTr.get_result();
     const DATA_IDX_TYPE sampled_time_signal_len = dataset6_len_ / dataset6_sample_period_;
     SparseDataContainer<DATA_REAL_VAL_TYPE>* sampled_time_signal_from_sparse_spectrum = new SparseDataContainer<DATA_REAL_VAL_TYPE>(); 
     for (int i = 0; i < dataset6_len_; i+=dataset6_sample_period_) {
	(*sampled_time_signal_from_sparse_spectrum).push_back({i,time_signal_from_sparse_spectrum[i][0]});
     }
     return std::unique_ptr<SparseDataContainer<DATA_REAL_VAL_TYPE>>(sampled_time_signal_from_sparse_spectrum);
  }

  std::unique_ptr<SparseDataContainer<DATA_REAL_VAL_TYPE>> prepare_dataset_7() {
     
     Inverse1DTransform ifftTr;
     auto& dataset7 = ifftTr.get_container(dataset7_len_);
     
     // dataset 7 - 5 non-zero frequencies + DC component, zeros everywhere else
     //
     dataset7[0][0]=1.0;
     dataset7[0][1]=0.0;
     dataset7[50][0]=0.0;
     dataset7[50][1]=0.0;

     dataset7[1][0] = 10;
     dataset7[1][1] = 10;

     dataset7[5][0] = 5;
     dataset7[5][1] = 1;

     dataset7[11][0] = 3;
     dataset7[11][1] = -2;

     dataset7[16][0] = 9;
     dataset7[16][1] = -1;

     dataset7[21][0] = 12;
     dataset7[21][1] = -0.5;

     dataset7[99][0] = 10;
     dataset7[99][1] = -10;

     dataset7[95][0] = 5;
     dataset7[95][1] = -1;

     dataset7[89][0] = 3;
     dataset7[89][1] = 2;

     dataset7[84][0] = 9;
     dataset7[84][1] = 1;

     dataset7[79][0] = 12;
     dataset7[79][1] = 0.5;

     LOG(INFO) << "Original Data: sumOfAbsYRe = " << dataset7.l1_norm()[0] << ", sumOfAbsYIm = " << dataset7.l1_norm()[1];

     ifftTr.execute();
     auto& time_signal_from_sparse_spectrum = ifftTr.get_result();
     const DATA_IDX_TYPE sampled_time_signal_len = dataset7_len_ / dataset7_sample_period_;
     SparseDataContainer<DATA_REAL_VAL_TYPE>* sampled_time_signal_from_sparse_spectrum = new SparseDataContainer<DATA_REAL_VAL_TYPE>(); 
     for (int i = 0; i < dataset7_len_; i+=dataset7_sample_period_) {
	(*sampled_time_signal_from_sparse_spectrum).push_back({i,time_signal_from_sparse_spectrum[i][0]});
     }
     return std::unique_ptr<SparseDataContainer<DATA_REAL_VAL_TYPE>>(sampled_time_signal_from_sparse_spectrum);
  }

  std::unique_ptr<SparseDataContainer<DATA_REAL_VAL_TYPE>> prepare_dataset_8() {
     // this is the result of dataset7
     //
     Inverse1DTransform ifftTr;
     auto& dataset8 = ifftTr.get_container(dataset8_len_);
     // sumOfAbsYRe = 79, sumOfAbsYIm = 6.66024
     // varX[0]=0, varY[0]=0
     // varX[5]=0, varY[5]=1.13339
     // varX[6]=0, varY[6]=2.19673
     // varX[11]=23.1341, varY[11]=0
     // varX[21]=0.129192, varY[21]=0
     // varX[22]=0.973622, varY[22]=0
     // varX[34]=0.179376, varY[34]=0
     // varX[35]=0.210669, varY[35]=0
     // varX[40]=5.2483, varY[40]=0
     // varX[41]=4.2263, varY[41]=0
     // varX[50]=10.7968, varY[50]=0
     // varX[59]=4.2263, varY[59]=0
     // varX[60]=5.2483, varY[60]=0
     // varX[65]=0.210669, varY[65]=0
     // varX[66]=0.179376, varY[66]=0
     // varX[78]=0.973622, varY[78]=0
     // varX[79]=0.129192, varY[79]=0
     // varX[89]=23.1341, varY[89]=0
     // varX[94]=0, varY[94]=-2.19673
     // varX[95]=0, varY[95]=-1.13339


     // dataset 8 - 9 non-zero frequencies + DC component, zeros everywhere else
     //
     dataset8[0][0]=0.0;
     dataset8[0][1]=0.0;
     dataset8[50][0]=10.7968;
     dataset8[50][1]=0.0;

     dataset8[5][0]=0;
     dataset8[5][1]=1.13339;

     dataset8[6][0]=0;
     dataset8[6][1]=2.19673;

     dataset8[11][0]=23.1341;
     dataset8[11][1]=0;

     dataset8[21][0]=0.129192;
     dataset8[21][1]=0;

     dataset8[22][0]=0.973622;
     dataset8[22][1]=0;

     dataset8[34][0]=0.179376;
     dataset8[34][1]=0;

     dataset8[35][1]=0.210669;
     dataset8[35][1]=0;

     dataset8[40][0]=5.2483;
     dataset8[40][1]=0;

     dataset8[41][0]=4.2263;
     dataset8[41][1]=0;

     dataset8[59][0]=4.2263;
     dataset8[59][1]=0;

     dataset8[60][0]=5.2483;
     dataset8[60][1]=0;

     dataset8[65][0]=0.210669;
     dataset8[65][1]=0;

     dataset8[66][0]=0.179376;
     dataset8[66][1]=0;

     dataset8[78][0]=0.973622;
     dataset8[78][1]=0;

     dataset8[79][0]=0.129192;
     dataset8[79][1]=0;

     dataset8[89][0]=23.1341;
     dataset8[89][1]=0;

     dataset8[94][0]=0;
     dataset8[94][1]=-2.19673;

     dataset8[95][0]=0;
     dataset8[95][1]=-1.13339;

     LOG(INFO) << "Original Data: sumOfAbsYRe = " << dataset8.l1_norm()[0] << ", sumOfAbsYIm = " << dataset8.l1_norm()[1];

     ifftTr.execute();
     auto& time_signal_from_sparse_spectrum = ifftTr.get_result();
     const DATA_IDX_TYPE sampled_time_signal_len = dataset8_len_ / dataset8_sample_period_;
     SparseDataContainer<DATA_REAL_VAL_TYPE>* sampled_time_signal_from_sparse_spectrum = new SparseDataContainer<DATA_REAL_VAL_TYPE>(); 
     for (int i = 0; i < dataset8_len_; i+=dataset8_sample_period_) {
	(*sampled_time_signal_from_sparse_spectrum).push_back({i,time_signal_from_sparse_spectrum[i][0]});
     }
     return std::unique_ptr<SparseDataContainer<DATA_REAL_VAL_TYPE>>(sampled_time_signal_from_sparse_spectrum);
  }

  void init_dataset_9(DenseDataContainer<DATA_COMPL_VAL_TYPE>& dataset, DATA_LEN_TYPE N) {
     dataset[0][0]=1.0;
     dataset[0][1]=0.0;
     dataset[N/2][0]=0.0;
     dataset[N/2][1]=0.0;

     dataset[N/100][0] = 10;
     dataset[N/100][1] = 10;

     dataset[7*N/100][0] = 5;
     dataset[7*N/100][1] = 1;

     dataset[15*N/100][0] = 3;
     dataset[15*N/100][1] = -2;

     dataset[23*N/100][0] = 9;
     dataset[23*N/100][1] = -1;

     dataset[31*N/100][0] = 12;
     dataset[31*N/100][1] = -0.5;

     dataset[N-N/100][0] = 10;
     dataset[N-N/100][1] = -10;

     dataset[N-7*N/100][0] = 5;
     dataset[N-7*N/100][1] = -1;

     dataset[N-15*N/100][0] = 3;
     dataset[N-15*N/100][1] = 2;

     dataset[N-23*N/100][0] = 9;
     dataset[N-23*N/100][1] = 1;

     dataset[N-31*N/100][0] = 12;
     dataset[N-31*N/100][1] = 0.5;

  }

  void prepare_dataset_9(int sample_size, SparseDataContainer<DATA_REAL_VAL_TYPE>& sampled_time_signal_from_sparse_spectrum) {
     
     // dataset 9 - 5 non-zero frequencies + DC component, zeros everywhere else
     //
     int d = dataset9_len_;
     int N = dataset9_len_;
     DenseDataContainer<DATA_COMPL_VAL_TYPE> dataset9(N);

     init_dataset_9(dataset9, dataset9_len_);   
 
     LOG(INFO) << "Original Data: sumOfAbsYRe = " << dataset9.l1_norm()[0] << ", sumOfAbsYIm = " << dataset9.l1_norm()[1];
     result9_ = std::unique_ptr<DenseDataContainer<DATA_REAL_VAL_TYPE>>(new DenseDataContainer<DATA_REAL_VAL_TYPE>(d));
     for (int i=0; i < d; ++i) {
       for (int j=0; j < d; ++j) {
          (*result9_)[i] += dataset9[j][0] * cos( ( 2.0 * M_PI * i * j ) / N ) - dataset9[j][1] * sin ( ( 2.0 * M_PI * i * j ) / N ) ; 
       }
     }

      
      /* initialize random seed: */
     srand (time(NULL));     
     std::set<int> ind; 
 
     int last = -1;
     std::cout << "Omega : ";
     for (int i  = 0; ind.size() < sample_size; ++i) {
        last = rand() % N;
        ind.insert(last);
     } 

     LOG(INFO) << "Signal Sample Size: " << ind.size();
     for (auto it = ind.begin(); it != ind.end(); ++it)  {
         std::cout << *it << ", ";
	     sampled_time_signal_from_sparse_spectrum.push_back({*it,(*result9_)[*it]});
     }
     std::cout << std::endl;
  }

  void prepare_datasets() {
     prepare_dataset_1();
     prepare_dataset_2();
     prepare_dataset_3();
     prepare_dataset_4();
  }


  SparseDataContainer<DATA_REAL_VAL_TYPE> dataset1_; // time domain dataset
  DATA_LEN_TYPE dataset1_len_;

  SparseDataContainer<DATA_REAL_VAL_TYPE> dataset2_; // time domain dataset
  DATA_LEN_TYPE dataset2_len_;

  SparseDataContainer<DATA_REAL_VAL_TYPE> dataset3_; // time domain dataset
  DATA_LEN_TYPE dataset3_len_;

  SparseDataContainer<DATA_REAL_VAL_TYPE> dataset4_; // time domain dataset
  DATA_LEN_TYPE dataset4_len_;

  DATA_LEN_TYPE dataset5_len_ = 100; 

  DATA_LEN_TYPE dataset5_sample_period_ = 10;

  DATA_LEN_TYPE dataset6_len_ = 100;

  DATA_LEN_TYPE dataset6_sample_period_ = 10;

  DATA_LEN_TYPE dataset7_len_ = 100;

  DATA_LEN_TYPE dataset7_sample_period_ = 10;

  DATA_LEN_TYPE dataset8_len_ = 100;

  DATA_LEN_TYPE dataset8_sample_period_ = 10;

  DATA_LEN_TYPE dataset9_len_ = 100;

  DATA_LEN_TYPE dataset9_sample_period_ = 10;

  DATA_LEN_TYPE dataset11_len_ = 100;

  DATA_LEN_TYPE dataset11_sample_period_ = 10;

  std::unordered_map<FourierForecaster::OptimizationSuite,DATA_REAL_VAL_TYPE> round_off_error_small_dataset_;
  std::unordered_map<FourierForecaster::OptimizationSuite,DATA_REAL_VAL_TYPE> round_off_error_large_dataset_;

  std::unique_ptr<DenseDataContainer<DATA_REAL_VAL_TYPE>> result9_;
 // Inverse1DTransform ifftTr11_; 

  // Objects declared here can be used by all tests in the test case for Foo.
};


// Test the DFT with various signals.
TEST_F(FourierForecasterTest, DISABLED_simpleForecastTest1) {
    FourierForecasterLinear ff("simpleForecasterTest1",FourierForecaster::GLOP);
    DATA_REAL_VAL_TYPE lambda = 0.1;
    ff.fit(dataset1_, dataset1_len_, lambda);
    auto& res = ff.get_result();
    const auto err = round_off_error_small_dataset_[ff.OptSuite()];
    EXPECT_TRUE(std::fabs(res[0] - 1.0) < err); 
    EXPECT_TRUE(std::fabs(res[1] - 1.0) < err);
    EXPECT_TRUE(std::fabs(res[2] - 0.0) < err);
    EXPECT_TRUE(std::fabs(res[3] - 0.0) < err);
}

TEST_F(FourierForecasterTest, DISABLED_simpleForecastTest2) {
    FourierForecasterLinear ff("simpleForecasterTest2",FourierForecaster::GLOP);
    DATA_REAL_VAL_TYPE lambda = 0.1;
    ff.fit(dataset2_, dataset2_len_, lambda);
    auto& res = ff.get_result();
    const auto err = round_off_error_small_dataset_[ff.OptSuite()];
    EXPECT_TRUE(std::fabs(res[0] - 2.0) < err); 
    EXPECT_TRUE(std::fabs(res[1] - 2.0) < err);
    EXPECT_TRUE(std::fabs(res[2] - 0.0) < err);
    EXPECT_TRUE(std::fabs(res[3] - 0.0) < err);
}

TEST_F(FourierForecasterTest, DISABLED_simpleForecastTest3) {
    FourierForecasterLinear ff("simpleForecasterTest3",FourierForecaster::GLOP);
    DATA_REAL_VAL_TYPE lambda = 0.1;
    ff.fit(dataset3_, dataset3_len_, lambda);
    auto& res = ff.get_result();
    const auto err = round_off_error_small_dataset_[ff.OptSuite()];
    EXPECT_TRUE(std::fabs(res[0] - 1.0) < err); 
    EXPECT_TRUE(std::fabs(res[1] - 0.0) < err);
    EXPECT_TRUE(std::fabs(res[2] - 1.0) < err);
    EXPECT_TRUE(std::fabs(res[3] - 0.0) < err);
}

// takes too long so it is disabled. Try it with SCIP solver
// -- dimitarpg 6-4-19 -- 
TEST_F(FourierForecasterTest, DISABLED_simpleForecastTest4) {
    FourierForecasterLinear ff("simpleForecasterTest4",FourierForecaster::GLOP);
    DATA_REAL_VAL_TYPE lambda = 0.1;
    ff.fit(dataset4_, dataset4_len_, lambda);
    auto& res = ff.get_result();
    const auto err = round_off_error_large_dataset_[ff.OptSuite()];
    EXPECT_TRUE(std::fabs(res[0] - 1.0) < err); 
    EXPECT_TRUE(std::fabs(res[1] - 0.0) < err);
    EXPECT_TRUE(std::fabs(res[50] - 0.0) < err);
    EXPECT_TRUE(std::fabs(res[99] - 1.0) < err);
}

TEST_F(FourierForecasterTest, DISABLED_subsampleInFreqDomainTest) {
    using operations_research::forecaster::Forward1DTransform;
    using operations_research::forecaster::Inverse1DTransform;

    Forward1DTransform fftTr;
    Inverse1DTransform ifftTr;

    fftTr.execute(dataset4_, dataset4_len_);
    const auto& res = fftTr.get_result();
    EXPECT_TRUE(fftTr.get_status()==FFT1DTransform::SUCCESS);

    int subsample_len = dataset4_len_/2+2; 
    int subsample_left_len = dataset4_len_/4;
    int subsample_right_len = (dataset4_len_*3)/4;
    auto& low_freq_sample = ifftTr.get_container(subsample_len);
    low_freq_sample[0][0] = res[0][0];
    low_freq_sample[0][1] = res[0][1];
  
    for (int i =1, j = subsample_left_len+2; i<=subsample_left_len; ++i)
    {
       low_freq_sample[i][0] = res[i][0];
       low_freq_sample[i][1] = res[i][1];
       low_freq_sample[j][0] = res[i+subsample_right_len-1][0];
       low_freq_sample[j][1] = res[i+subsample_right_len-1][1];
       ++j;
    }
    low_freq_sample[subsample_left_len+1][0]=0.0;
    low_freq_sample[subsample_left_len+1][1]=0.0;

    ifftTr.execute(); 
    const auto& low_freq_time_signal = ifftTr.get_result();
    //for (int i=0; i<subsample_len; ++i) {
    //   std::cout << i << ":  " << low_freq_time_signal[i][0] / DATA_REAL_VAL_TYPE(subsample_len-2) << std::endl;
    //}

    EXPECT_TRUE(ifftTr.get_status() == FFT1DTransform::SUCCESS);
    const DATA_REAL_VAL_TYPE sample_err = 1e-5;
    const DATA_REAL_VAL_TYPE sample0 = 0.998421;
    EXPECT_TRUE(std::fabs(low_freq_time_signal[0][0] / DATA_REAL_VAL_TYPE(subsample_len-2) - sample0) < sample_err);
    const DATA_REAL_VAL_TYPE sample50 = 0.982373;
    EXPECT_TRUE(std::fabs(low_freq_time_signal[50][0] / DATA_REAL_VAL_TYPE(subsample_len-2) - sample50) < sample_err);
    const DATA_REAL_VAL_TYPE max_sample_magnitude = 0.2;
    for (int i = 1; i < subsample_len - 1; ++i) {
       if (i != 50) {
           EXPECT_TRUE(std::fabs(low_freq_time_signal[i][0] / DATA_REAL_VAL_TYPE(subsample_len-2)) < max_sample_magnitude);
       }
    } 
    const DATA_REAL_VAL_TYPE max_imag_magnitude = 1e-10;
    for (int i = 0; i < subsample_len - 1; ++i) {
       EXPECT_TRUE(std::fabs(low_freq_time_signal[i][1] / DATA_REAL_VAL_TYPE(subsample_len-2)) < max_imag_magnitude);
    } 
}

TEST_F(FourierForecasterTest, DISABLED_superresolutionTest1) {
    using operations_research::forecaster::Forward1DTransform;
    using operations_research::forecaster::Inverse1DTransform;
    // construct frequency spectrum
    auto sparse_time_signal = prepare_dataset_5();
    FourierForecasterLinear ff("superresolutionTest1",FourierForecaster::SCIP);
    DATA_REAL_VAL_TYPE lambda = 1; 
    ff.fit(*sparse_time_signal, dataset5_len_, lambda);
    auto& res = ff.get_result();
    

    EXPECT_TRUE(true);
}

TEST_F(FourierForecasterTest, DISABLED_superresolutionTest2) {
    using operations_research::forecaster::Forward1DTransform;
    using operations_research::forecaster::Inverse1DTransform;
    // construct frequency spectrum
    auto sparse_time_signal = prepare_dataset_6();
    FourierForecasterLinear ff("superresolutionTest1",FourierForecaster::SCIP);
    DATA_REAL_VAL_TYPE lambda = 1; 
    ff.fit(*sparse_time_signal, dataset6_len_, lambda);
    auto& res = ff.get_result();
    

    EXPECT_TRUE(true);
}
 
TEST_F(FourierForecasterTest, superresolutionTest3) {
    using operations_research::forecaster::Forward1DTransform;
    using operations_research::forecaster::Inverse1DTransform;
    // construct frequency spectrum
    std::vector<DATA_REAL_VAL_TYPE> lambdas = {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0  };
    std::vector<DATA_IDX_TYPE> sizes = { 7, 10, 12, 15, 20, 25, 30, 50, 100  };
    DATA_LEN_TYPE subsample_count = DATA_LEN_TYPE(10);
    DATA_REAL_VAL_TYPE recoveredSparsity[lambdas.size()][sizes.size()];
    DATA_REAL_VAL_TYPE reconstructError[lambdas.size()][sizes.size()];
    DATA_REAL_VAL_TYPE overallReconstructError[lambdas.size()][sizes.size()];
    DATA_REAL_VAL_TYPE sumAbsYRe[lambdas.size()][sizes.size()];
    DATA_REAL_VAL_TYPE sumAbsYIm[lambdas.size()][sizes.size()];
    DATA_REAL_VAL_TYPE l1Norm[lambdas.size()][sizes.size()];

    for (int l = 0; l < lambdas.size(); ++l) {
       for (int s = 0; s < sizes.size(); ++s) {
          DATA_REAL_VAL_TYPE AvgRecoveredSpectralSparsity = DATA_REAL_VAL_TYPE(0);
          DATA_REAL_VAL_TYPE AvgReconstructPercentError = DATA_REAL_VAL_TYPE(0); 
          DATA_REAL_VAL_TYPE AvgOverallReconstructPercentError = DATA_REAL_VAL_TYPE(0);
          DATA_REAL_VAL_TYPE AvgSumAbsYRe = DATA_REAL_VAL_TYPE(0);
          DATA_REAL_VAL_TYPE AvgSumAbsYIm = DATA_REAL_VAL_TYPE(0);
          DATA_REAL_VAL_TYPE AvgL1Norm = DATA_REAL_VAL_TYPE(0);
          for (int iter = 0; iter < subsample_count; ++iter)   {
            SparseDataContainer<DATA_REAL_VAL_TYPE> sparse_time_signal;
            prepare_dataset_9(sizes[s],sparse_time_signal);
            std::unique_ptr<FourierForecasterLinear> ff = std::unique_ptr<FourierForecasterLinear>(new FourierForecasterLinear("superresolutionTest1",FourierForecaster::SCIP));
            ff->fit(sparse_time_signal, dataset9_len_, lambdas[l]);
            auto& res1 = ff->get_result();
            auto& res2 = *result9_;
            DATA_REAL_VAL_TYPE err_abs=DATA_REAL_VAL_TYPE(0), sumRe=DATA_REAL_VAL_TYPE(0);
            std::unordered_set<int> ind;
            for (int i = 0; i < sparse_time_signal.size(); ++i)
            {
               ind.insert(sparse_time_signal[i].first);
            }
            std::vector<double> filtered;
            for (int i = 0 ; i < dataset9_len_; ++i) {
                if (ind.find(i) == ind.end()) {
                   filtered.push_back(res2[i]);
                }
            }
            int leftoverCount = dataset9_len_ - sparse_time_signal.size();
            for (int i = 0; i < leftoverCount; ++i) {
              err_abs += std::abs(filtered[i]-res1[i+10]);
              sumRe += std::abs(filtered[i]);
            }
            for (int i = 0; i < sparse_time_signal.size(); ++i) {
               err_abs += std::abs(res1[i] - sparse_time_signal[i].second);
               sumRe += std::abs(sparse_time_signal[i].second);
            }
            DATA_REAL_VAL_TYPE overallPercentError = err_abs/sumRe*100.0;
            LOG(INFO) << "Time domain error against true signal: " << overallPercentError;

            AvgOverallReconstructPercentError += overallPercentError;
            AvgReconstructPercentError += ff->percentErr_;
            AvgRecoveredSpectralSparsity += ff->recoveredSparsity_;
            AvgSumAbsYRe += ff->freqNorm_.first;
            AvgSumAbsYIm += ff->freqNorm_.second;
            AvgL1Norm += ff->l1_norm_;
          }
          recoveredSparsity[l][s] = AvgRecoveredSpectralSparsity / ((DATA_REAL_VAL_TYPE) subsample_count);
          reconstructError[l][s] = AvgReconstructPercentError / ((DATA_REAL_VAL_TYPE) subsample_count);
          overallReconstructError[l][s] = AvgOverallReconstructPercentError / ((DATA_REAL_VAL_TYPE) subsample_count);
          sumAbsYRe[l][s] = AvgSumAbsYRe / ((DATA_REAL_VAL_TYPE) subsample_count);
          sumAbsYIm[l][s] = AvgSumAbsYIm / ((DATA_REAL_VAL_TYPE) subsample_count);
          l1Norm[l][s] = AvgL1Norm /  ((DATA_REAL_VAL_TYPE) subsample_count);
        
       }
    }


   std::cout << "Lambda, |Omega|, RecoveredSpectralSparsity, ReconstructedPercentError, OverallReconstructedPercentError, sum |y_{re}|, sum |y_{im}|, l1 norm" << std::endl;
   for (int l = 0; l < lambdas.size(); ++l) {
      for (int s = 0; s < sizes.size(); ++s) {
         std::cout << lambdas[l] << ", " << sizes[s] << ", " 
                   << recoveredSparsity[l][s] << ", " << reconstructError[l][s] << ", " 
                   << overallReconstructError[l][s] << ", " << sumAbsYRe[l][s] << ", " 
                   << sumAbsYIm[l][s] << ", " << l1Norm[l][s] << std::endl;
      }

   }     
    

    EXPECT_TRUE(true);
}

} // empty namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
