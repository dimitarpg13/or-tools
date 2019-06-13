#ifndef OR_TOOLS_FORECASTER_FOURIER_1D_H_
#define OR_TOOLS_FORECASTER_FOURIER_1D_H_
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <unordered_map>
#include <complex.h>
#include <type_traits>
#include <limits>
#include "ortools/linear_solver/linear_solver.h"
#include "fftw3.h"

namespace operations_research {
namespace forecaster {

typedef int DATA_LEN_TYPE; 
typedef int DATA_IDX_TYPE;
typedef std::decay<std::remove_extent<fftw_complex>::type>::type DATA_REAL_VAL_TYPE;
typedef fftw_complex DATA_COMPL_VAL_TYPE;

template<typename T>
struct C__
{
   typedef std::pair<DATA_IDX_TYPE,T> SparseDataElement;
};

template<typename T>
using SparseDataElement = typename C__<T>::SparseDataElement;

template<typename T>
using SparseDataContainer = typename std::vector<SparseDataElement<T>>; 

template<typename T>
using ConstSparseDataContainer = typename std::vector<SparseDataElement<T>> const;


// sparse data encoding (SDE) adopted here is kind of run-length encoding:
// a list of Key-Value pairs ordered by Key in increasing order.
// Key is of type DATA_IDX_TYPE, Value is of type DATA_REAL_VAL_TYPE
// or DATA_COMPL_VAL_TYPE. The Key indicates the offset of the current
// value along the axis grid. If there are two Keys with the same value
// then the value of second Value will indicate the number of repetitions 
// of the Value which corresponds to the previous Key.
// For instance the sequence:
// (3, 10.1), (5, 13.7), (5, 10.0), (16, 17.6)
// represents the following grid of values:
// 0.0,0.0,0.0,10.1,0.0,13.7,13.7,13.7,13.7,13.7,13.7,13.7,13.7,13.7,13.7,0.0,17.6
 
// Note: in our forecaster problem we are going to use real valued signals in time domain
// which implies that the DFT of the signal will be an array of complex numbers where
// the first half of the array is complex conjugate of the second half. If the array is of 
// length N and the frequency domain array values are denoted by y_r[k] + i* y_i[k], 0 <= k <= N/2
// then the inverse DFT x[l], 0 <= l <= N-1 is given with:
// x[l] = \sum_{k=0}^{\frac{N}{2}} y_r[k] {\cos}(\frac{2{\pi}kl}{N}) - y_i[l] {\sin}(\frac{2{\pi}kl}{N})  

class FFT1DTransform;

// first member contains the absolute error and the second member contains 
// the relative error
typedef std::pair<DATA_REAL_VAL_TYPE, DATA_REAL_VAL_TYPE> DatasetError;

template<typename T>
struct DenseDataContainer {
    DenseDataContainer() = delete;
    DenseDataContainer(const DenseDataContainer& ) = delete;
    DenseDataContainer(DenseDataContainer&& ) = default;
    DenseDataContainer(const DATA_LEN_TYPE);
    ~DenseDataContainer();
    const T* const* data();
    const DATA_LEN_TYPE size() const;
    const T& operator[] (const DATA_IDX_TYPE&) const;
    const T& NaN() { return NaN_; };
    const T& l1_norm(); 
    void error(const DenseDataContainer<T>& , DatasetError & );
    void error(const SparseDataContainer<T>& , const std::vector<DATA_IDX_TYPE>& , DatasetError & );
    T& operator[] (const DATA_IDX_TYPE&);
    friend class FFT1DTransform;
    friend class Forward1DTransform;
    friend class Inverse1DTransform;
  private:
    bool l1_is_old_;
    T NaN_;
    T l1_norm_;
    T* data_;
    DATA_LEN_TYPE N_;
};

template<>
struct DenseDataContainer<DATA_COMPL_VAL_TYPE> {
    DenseDataContainer() = delete;
    DenseDataContainer(const DenseDataContainer& ) = delete;
    DenseDataContainer(DenseDataContainer&& ) = default;
    DenseDataContainer(const DATA_LEN_TYPE); 
    const DATA_COMPL_VAL_TYPE * const* data();
    const DATA_LEN_TYPE size() const;
    const DATA_COMPL_VAL_TYPE& operator[] (const DATA_IDX_TYPE&) const;
    ~DenseDataContainer();
    const DATA_COMPL_VAL_TYPE& NaN() { return NaN_; };
    const DATA_COMPL_VAL_TYPE& l1_norm();
    void error(const DenseDataContainer<DATA_COMPL_VAL_TYPE>& , DatasetError & );
    void error(const SparseDataContainer<DATA_COMPL_VAL_TYPE>& , const std::vector<DATA_IDX_TYPE>& , DatasetError & );
    DATA_COMPL_VAL_TYPE& operator[] (const DATA_IDX_TYPE&);
    friend class FFT1DTransform;
    friend class Forward1DTransform;
    friend class Inverse1DTransform;
  private:
    bool l1_is_old_;
    DATA_COMPL_VAL_TYPE NaN_;
    DATA_COMPL_VAL_TYPE l1_norm_;
    DATA_COMPL_VAL_TYPE* data_;
    DATA_LEN_TYPE N_;
};

class FFT1DTransform {
  public:
    enum TransformStatus {
      SUCCESS=0,
      MISSING_DATA=1,
      INCORRECT_DATA_FORMAT=2,
      TRUNCATED_DATA=4,
      INTERNAL_ERROR=8,
      UNSPECIFIED=16
    };
    FFT1DTransform();
    ~FFT1DTransform() { 
      clear(); 
     };
    void execute(ConstSparseDataContainer<DATA_REAL_VAL_TYPE> & , DATA_LEN_TYPE N);
    void execute(ConstSparseDataContainer<DATA_COMPL_VAL_TYPE> & , DATA_LEN_TYPE N);

    DenseDataContainer<DATA_COMPL_VAL_TYPE>& get_container(const DATA_LEN_TYPE N);
    void execute();
    DenseDataContainer<DATA_COMPL_VAL_TYPE>& get_result();
    const int get_status();
    void clear(); 
    template<typename T>
    friend class DenseDataContainer;
  protected:
    void init(DATA_LEN_TYPE N);
    void populate_from_sparse(ConstSparseDataContainer<DATA_REAL_VAL_TYPE> & data, DATA_LEN_TYPE N);
    void populate_from_sparse(ConstSparseDataContainer<DATA_COMPL_VAL_TYPE> & data, DATA_LEN_TYPE N);
    
    template<typename T>
    void execute_sparse(T& data, DATA_LEN_TYPE N) {
       init(N);
       plan();
       populate_from_sparse(data,N);
       fftw_execute(plan_);
       need_to_clear_ = true;
    }

    virtual void plan() = 0;

    DATA_LEN_TYPE N_;
    std::unique_ptr<DenseDataContainer<DATA_COMPL_VAL_TYPE>> in_;
    std::unique_ptr<DenseDataContainer<DATA_COMPL_VAL_TYPE>> out_; 

    fftw_plan plan_;
    int status_;
    bool need_to_clear_;
};

class Forward1DTransform : public FFT1DTransform {
  public:
      Forward1DTransform();
      ~Forward1DTransform();
      template<typename T>
      friend class DenseDataContainer;
  protected:
      void plan() override;
};

class Inverse1DTransform : public FFT1DTransform {
  public:
      Inverse1DTransform();
      ~Inverse1DTransform();
      template<typename T>
      friend class DenseDataContainer;
  protected:
      void plan() override;
};

template struct DenseDataContainer<DATA_REAL_VAL_TYPE>;
template struct DenseDataContainer<DATA_COMPL_VAL_TYPE>;

} // ns: forecaster
} // ns: operations_research

#endif // OR_TOOLS_FORECASTER_FOURIER_FORECASTER_H_

