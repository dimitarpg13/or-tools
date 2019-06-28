#include <string>
#include <unordered_map>
#include <iostream>
#include "gtest/gtest.h"
#include "ortools/forecaster/fourier_1d.h"
#include "ortools/forecaster/fourier_forecaster.h"


namespace {
 using namespace operations_research::forecaster;

// The fixture for testing class Foo.
class FourierDFTTest : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  FourierDFTTest() {
     // You can do set-up work for each test here.
  }

  ~FourierDFTTest() override {
     // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
     prepare_datasets();
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }

  void prepare_dataset_1() {

     // dataset 1 - simple test example with 4 points
     // x = [ 1, 2-i, -i, -1+2i ]
     //
     dataset1_len_ = 4;
     
     SparseDataElement<DATA_COMPL_VAL_TYPE> e1;
     e1.first = 0;
     e1.second[0] = 1.0;
     e1.second[1] = 0.0;
     dataset1_.push_back(std::move(e1));
     
     SparseDataElement<DATA_COMPL_VAL_TYPE> e2;
     e2.first = 1;
     e2.second[0] = 2.0;
     e2.second[1] = -1.0;
     dataset1_.push_back(std::move(e2));
     
     SparseDataElement<DATA_COMPL_VAL_TYPE> e3;
     e3.first = 2;
     e3.second[0] = 0.0;
     e3.second[1] = -1.0;
     dataset1_.push_back(std::move(e3));

     SparseDataElement<DATA_COMPL_VAL_TYPE> e4;
     e4.first = 3;
     e4.second[0] = -1.0;
     e4.second[1] = 2.0;
     dataset1_.push_back(std::move(e4));



  }

  void prepare_dataset_2() {

     // dataset 1 - the inverse of dataset 1
     // x = [ 1, 2-i, -i, -1+2i ]
     //
     dataset2_len_ = 4;
     
     SparseDataElement<DATA_COMPL_VAL_TYPE> e1;
     e1.first = 0;
     e1.second[0] = 2.0;
     e1.second[1] = 0.0;
     dataset2_.push_back(std::move(e1));
     
     SparseDataElement<DATA_COMPL_VAL_TYPE> e2;
     e2.first = 1;
     e2.second[0] = -2.0;
     e2.second[1] = -2.0;
     dataset2_.push_back(std::move(e2));
     
     SparseDataElement<DATA_COMPL_VAL_TYPE> e3;
     e3.first = 2;
     e3.second[0] = 0.0;
     e3.second[1] = -2.0;
     dataset2_.push_back(std::move(e3));

     SparseDataElement<DATA_COMPL_VAL_TYPE> e4;
     e4.first = 3;
     e4.second[0] = 4.0;
     e4.second[1] = 4.0;
     dataset2_.push_back(std::move(e4));

  }

  void prepare_datasets() {
     prepare_dataset_1();
     prepare_dataset_2();

  }

  SparseDataContainer<DATA_COMPL_VAL_TYPE> dataset1_;
  DATA_LEN_TYPE dataset1_len_;

  SparseDataContainer<DATA_COMPL_VAL_TYPE> dataset2_;
  DATA_LEN_TYPE dataset2_len_;
  // Objects declared here can be used by all tests in the test case for Foo.
};


// Test the DFT with various signals.
TEST_F(FourierDFTTest, forwardDFT1DSparseTest) {
    using operations_research::forecaster::Forward1DTransform;
    Forward1DTransform fftTr;
    fftTr.execute(dataset1_, dataset1_len_);
    const auto& res = fftTr.get_result();
    EXPECT_EQ(res[0][0],2.0);
    EXPECT_EQ(res[0][1],0.0);
    EXPECT_EQ(res[1][0],-2.0);
    EXPECT_EQ(res[1][1],-2.0);
    EXPECT_EQ(res[2][0],0.0);
    EXPECT_EQ(res[2][1],-2.0);
    EXPECT_EQ(res[3][0],4.0);
    EXPECT_EQ(res[3][1],4.0);
    const auto& st = fftTr.get_status(); 
    EXPECT_TRUE(st == FFT1DTransform::TransformStatus::SUCCESS);
}

// x = [ 1, 2-i, -i, -1+2i ]
TEST_F(FourierDFTTest, inverseDFT1DSparseTest) {
    using operations_research::forecaster::Inverse1DTransform;
    Inverse1DTransform ifftTr;
    ifftTr.execute(dataset2_, dataset2_len_); 
    const auto& res = ifftTr.get_result();
    EXPECT_EQ(res[0][0] / dataset2_len_, 1);
    EXPECT_EQ(res[0][1] / dataset2_len_, 0);
    EXPECT_EQ(res[1][0] / dataset2_len_, 2);
    EXPECT_EQ(res[1][1] / dataset2_len_, -1);
    EXPECT_EQ(res[2][0] / dataset2_len_, 0);
    EXPECT_EQ(res[2][1] / dataset2_len_, -1);
    EXPECT_EQ(res[3][0] / dataset2_len_, -1);
    EXPECT_EQ(res[3][1] / dataset2_len_, 2);
    const auto& st = ifftTr.get_status(); 
    EXPECT_TRUE(st == FFT1DTransform::TransformStatus::SUCCESS);
    EXPECT_TRUE(true);
}

// dataset 1 - the inverse of dataset 1
// x = [ 1, 2-i, -i, -1+2i ]
TEST_F(FourierDFTTest, forwardDFT1DDenseTest) {
    using operations_research::forecaster::Forward1DTransform;
    Forward1DTransform fftTr;
    auto& c = fftTr.get_container(dataset1_len_);
    c[0][0] = 1;
    c[0][1] = 0; 
    c[1][0] = 2;
    c[1][1] = -1;
    c[2][0] = 0;
    c[2][1] = -1;
    c[3][0] = -1;
    c[3][1] = 2;
    fftTr.execute();
    const auto& res = fftTr.get_result();
    EXPECT_EQ(res[0][0],2.0);
    EXPECT_EQ(res[0][1],0.0);
    EXPECT_EQ(res[1][0],-2.0);
    EXPECT_EQ(res[1][1],-2.0);
    EXPECT_EQ(res[2][0],0.0);
    EXPECT_EQ(res[2][1],-2.0);
    EXPECT_EQ(res[3][0],4.0);
    EXPECT_EQ(res[3][1],4.0);
    const auto& st = fftTr.get_status(); 
    EXPECT_TRUE(st == FFT1DTransform::TransformStatus::SUCCESS);
}

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
