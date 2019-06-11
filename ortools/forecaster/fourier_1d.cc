#include "ortools/forecaster/fourier_1d.h"

namespace operations_research {
namespace forecaster {

DenseDataContainer<DATA_COMPL_VAL_TYPE>::DenseDataContainer(const DATA_LEN_TYPE N) : 
	N_(N), l1_is_old_(false) {
   if (N > 0)
     data_ = (DATA_COMPL_VAL_TYPE *) fftw_malloc(sizeof(DATA_COMPL_VAL_TYPE) * N);
   else
     data_ = nullptr;
   NaN_[0] =  std::numeric_limits<DATA_REAL_VAL_TYPE>::quiet_NaN();
   NaN_[1] =  std::numeric_limits<DATA_REAL_VAL_TYPE>::quiet_NaN();

}

const DATA_COMPL_VAL_TYPE& DenseDataContainer<DATA_COMPL_VAL_TYPE>::operator[](const DATA_IDX_TYPE& idx) const {
   if (idx < N_) {
       return data_[idx];
   }
   else
   {
       //TODO (dpg): maybe we should throw instead of returning
       return NaN_;
   } 
}

const DATA_COMPL_VAL_TYPE* const* DenseDataContainer<DATA_COMPL_VAL_TYPE>::data()  {
   return &data_;
}

const DATA_COMPL_VAL_TYPE& DenseDataContainer<DATA_COMPL_VAL_TYPE>::l1_norm() 
{
   if (!l1_is_old_) {
       return l1_norm_;
   }
   else
   {
      l1_norm_[0] = DATA_REAL_VAL_TYPE(0);
      l1_norm_[1] = DATA_REAL_VAL_TYPE(1);
      for (int i = 0; i < N_; ++i) {
        l1_norm_[0] += std::fabs(data_[i][0]);
        l1_norm_[1] += std::fabs(data_[i][1]);
      }
      l1_is_old_=false;
   } 
};

void DenseDataContainer<DATA_COMPL_VAL_TYPE>::error(const DenseDataContainer<DATA_COMPL_VAL_TYPE>& other, DatasetError& err) {
  DATA_LEN_TYPE size = std::min(N_, other.size());
  DATA_REAL_VAL_TYPE errAbs(0), errPerc(0), sumOfAbs(0);
  for (DATA_LEN_TYPE i = 0; i < size; ++i) {
      //TO DO (dpg): it has to be sqrt of the squares
      errAbs += fabs(data_[i][0] - other[i][0]);
      errAbs += fabs(data_[i][1] - other[i][1]);
      sumOfAbs += fabs(other[i][0]);
      sumOfAbs += fabs(other[i][1]); 
  }
}

void DenseDataContainer<DATA_COMPL_VAL_TYPE>::error(const SparseDataContainer<DATA_COMPL_VAL_TYPE>& other, DatasetError& err) {
   DATA_LEN_TYPE S = other.size();
   DATA_REAL_VAL_TYPE errAbs(0), errPerc(0), sumOfAbsOther(0);
   for (DATA_LEN_TYPE i = 0; i < S; ++i) {
      //TO DO (dpg): it has to be sqrt of the squares
      errAbs += fabs(data_[other[i].first][0] - other[i].second[0]);
      errAbs += fabs(data_[other[i].first][1] - other[i].second[1]);
      sumOfAbsOther += fabs(other[i].second[0]); 
      sumOfAbsOther += fabs(other[i].second[1]);
   }
   err.first = errAbs;
   err.second = errAbs/sumOfAbsOther*100.0; 
}

const DATA_LEN_TYPE DenseDataContainer<DATA_COMPL_VAL_TYPE>::size() const {
   return N_;
}   

DATA_COMPL_VAL_TYPE& DenseDataContainer<DATA_COMPL_VAL_TYPE>::operator[](const DATA_IDX_TYPE& idx) {
   l1_is_old_=true;
   return data_[idx];
}

DenseDataContainer<DATA_COMPL_VAL_TYPE>::~DenseDataContainer() {
  if (data_ != nullptr)
  {
     fftw_free(data_);
     data_ = nullptr;
     N_ = 0;
  } 
}

template<typename T>
DenseDataContainer<T>::DenseDataContainer(const DATA_LEN_TYPE N) : N_(N), l1_is_old_(false) {
   if (N > 0)
     data_ = (T *) calloc(N, sizeof(T));
   else
     data_ = nullptr;
   NaN_ =  std::numeric_limits<T>::quiet_NaN();
}

template<typename T>
DenseDataContainer<T>::~DenseDataContainer() {
  if (data_ != nullptr)
  {
     free(data_);
     data_ = nullptr;
     N_ = 0;
  } 
}

template<typename T>
const T& DenseDataContainer<T>::operator[](const DATA_IDX_TYPE& idx) const {
   if (idx < N_) {
       return data_[idx];
   }
   else
   {
       //TODO (dpg): maybe we should throw instead of returning
       return NaN_;
   } 
}

template<typename T>
const T* const* DenseDataContainer<T>::data() {
   return &data_;
}

template<typename T>
const DATA_LEN_TYPE DenseDataContainer<T>::size() const {
   return N_;
}   

template<typename T>
T& DenseDataContainer<T>::operator[](const DATA_IDX_TYPE& idx) {
   l1_is_old_ = true;
   return data_[idx];
}

template<typename T>
const T& DenseDataContainer<T>::l1_norm() 
{
   if (!l1_is_old_) {
       return l1_norm_;
   }
   else
   {
      l1_norm_ = T(0);
      for (int i = 0; i < N_; ++i) {
        l1_norm_ += std::fabs(data_[i]);
      }
      l1_is_old_=false;
   } 
}; 

template<typename T>
void DenseDataContainer<T>::error(const DenseDataContainer<T>& other, DatasetError& err) {
  DATA_LEN_TYPE size = std::min(N_, other.size());
  DATA_REAL_VAL_TYPE errAbs(0), errPerc(0), sumOfAbs(0);
  for (DATA_LEN_TYPE i = 0; i < size; ++i) {
      errAbs += fabs(data_[i] - other[i]);
      sumOfAbs += fabs(other[i]);
  }
}

template<typename T>
void DenseDataContainer<T>::error(const SparseDataContainer<T>& other, DatasetError& err) {
   DATA_LEN_TYPE S = other.size();
   DATA_REAL_VAL_TYPE errAbs(0), errPerc(0), sumOfAbsOther(0);
   for (DATA_LEN_TYPE i = 0; i < S; ++i) {
      //TO DO (dpg): it has to be sqrt of the squares
      errAbs += fabs(data_[other[i].first] - other[i].second);
      errAbs += fabs(data_[other[i].first] - other[i].second);
      sumOfAbsOther += fabs(other[i].second); 
      sumOfAbsOther += fabs(other[i].second);
   }
   err.first = errAbs;
   err.second = errAbs/sumOfAbsOther*100.0; 
}

FFT1DTransform::FFT1DTransform() :
   need_to_clear_(false), in_(nullptr), out_(nullptr), status_(SUCCESS) 
{}

void FFT1DTransform::init(DATA_LEN_TYPE N) {
   N_ = N;
   in_ = std::unique_ptr<DenseDataContainer<DATA_COMPL_VAL_TYPE>>(new DenseDataContainer<DATA_COMPL_VAL_TYPE>(N));
   out_ = std::unique_ptr<DenseDataContainer<DATA_COMPL_VAL_TYPE>>(new DenseDataContainer<DATA_COMPL_VAL_TYPE>(N));
}

void FFT1DTransform::execute(ConstSparseDataContainer<DATA_REAL_VAL_TYPE> & data, DATA_LEN_TYPE N) {
   execute_sparse(data, N);
}

void FFT1DTransform::execute(ConstSparseDataContainer<DATA_COMPL_VAL_TYPE> & data, DATA_LEN_TYPE N) {
   execute_sparse(data, N);
}

DenseDataContainer<DATA_COMPL_VAL_TYPE>& FFT1DTransform::get_container(const DATA_LEN_TYPE N) { 
   init(N);
   plan();
   return *in_;
}

void FFT1DTransform::execute() {
   fftw_execute(plan_);
   need_to_clear_ = true; 
}

void FFT1DTransform::populate_from_sparse(ConstSparseDataContainer<DATA_REAL_VAL_TYPE> & data, DATA_LEN_TYPE N) {
   DATA_IDX_TYPE i(0);
   for (auto it : data) {
      const auto& idx = it.first;
      if (idx > i) {
         // fill in all elements in the last gap i.e. between i..idx-1
         for (; i < idx; ++i) {
            in_->data_[i][0] = DATA_REAL_VAL_TYPE(0);
            in_->data_[i][1] = DATA_REAL_VAL_TYPE(0);
         }
      }
      else if (idx == i - 1) {
	 const auto& val = (*in_)[i-1][0];
         // fill in all repetitions for the current value it.second
	 for (int j = 0; j < it.second; ++j) {
            in_->data_[i][0] = val;
	    in_->data_[i][1] = DATA_REAL_VAL_TYPE(0); 
	    ++i;
	 } 
      } 
      else if (i > idx + 1) {
          status_ |= INCORRECT_DATA_FORMAT;
      }
      in_->data_[i][0] = it.second;
      in_->data_[i][1] = DATA_REAL_VAL_TYPE(0);
      ++i;
      if (i > N) {
         status_ |= TRUNCATED_DATA;
      }
   }
   if (i < N) {
      const auto& val = in_->data_[i-1][0];
      for (; i < N; ++i) {
         in_->data_[i][0] = val;
	 in_->data_[i][1] = DATA_REAL_VAL_TYPE(0);
      }
   }
};

void FFT1DTransform::populate_from_sparse(ConstSparseDataContainer<DATA_COMPL_VAL_TYPE> & data, DATA_LEN_TYPE N) {
   DATA_IDX_TYPE i(0);
   for (auto it : data) {
      const auto& idx = it.first;
      if (idx > i) {
         // fill in all elements in the last gap i.e. between i..idx-1
         for (; i < idx; ++i) {
            in_->data_[i][0] = DATA_REAL_VAL_TYPE(0);
            in_->data_[i][1] = DATA_REAL_VAL_TYPE(0);
         }
      }
      else if (idx == i - 1) {
	 const auto& real = in_->data_[i-1][0];
	 const auto& imag = in_->data_[i-1][1];
         // fill in all repetitions for the current value it.second
	 for (DATA_IDX_TYPE j = 0; j < it.second[0]; ++j) {
            in_->data_[i][0] = real;
	    in_->data_[i][1] = imag; 
	    ++i;
	 } 
      }
      else if (i > idx + 1) {
          status_ |= INCORRECT_DATA_FORMAT;
      }
      in_->data_[i][0] = it.second[0];
      in_->data_[i][1] = it.second[1];
      ++i;
      if (i > N) {
         status_ |= TRUNCATED_DATA;
      }
   }
   if (i < N) {
      const auto& real = in_->data_[i-1][0];
      const auto& imag = in_->data_[i-1][1];
      for (; i < N; ++i) {
         in_->data_[i][0] = real;
	 in_->data_[i][1] = imag;
      }
   }
};

void FFT1DTransform::clear() {
   fftw_destroy_plan(plan_);
   in_ = nullptr;
   out_ = nullptr; 
   N_ = 0;
   need_to_clear_ = false;
}

DenseDataContainer<DATA_COMPL_VAL_TYPE>& FFT1DTransform::get_result() {
   return *out_;
}

const int FFT1DTransform::get_status() {
   return status_;
}

//TODO (dpg): 
Forward1DTransform::Forward1DTransform()  
{

}

//TODO (dpg): 
Forward1DTransform::~Forward1DTransform() {
   clear();
}

void Forward1DTransform::plan() {
   //TODO (dpg): execute here
   //
   if (N_ > 0)
      plan_ = fftw_plan_dft_1d(N_, in_->data_, out_->data_, FFTW_FORWARD, FFTW_MEASURE);
   else
      status_ |= MISSING_DATA;
}

//TODO (dpg): 
Inverse1DTransform::Inverse1DTransform() {

}

//TODO (dpg):
Inverse1DTransform::~Inverse1DTransform() {

}

void Inverse1DTransform::plan() {
   //TODO (dpg): execute here
   //
   if (N_ > 0)
       plan_ = fftw_plan_dft_1d(N_, in_->data_, out_->data_, FFTW_BACKWARD, FFTW_MEASURE);
   else
       status_ |= MISSING_DATA;
}

} // ns: forecaster

} // ns: operations_research   
