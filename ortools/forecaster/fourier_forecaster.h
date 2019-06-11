#ifndef OR_TOOLS_FORECASTER_FOURIER_FORECASTER_H_
#define OR_TOOLS_FORECASTER_FOURTIER_FORECASTER_H_
#include <vector>
#include <string>
#include <utility>
#include <math.h>
#include <memory>
#include <unordered_map>
#include <complex.h>
#include <type_traits>
#include "ortools/base/timer.h"
#include <limits>
#include "ortools/linear_solver/linear_solver.h"
#include "forecaster.h"
#include "fourier_1d.h"

namespace operations_research {
namespace forecaster {

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

class FourierForecaster : public Forecaster {
  public:
     enum OptimizationSuite {
#ifdef USE_GUROBI
       GUROBI = 1, // Recommended default value
#endif
#ifdef USE_GLPK
       GLPK = 2,  
#endif
#ifdef USE_CLP
       CLP = 4,
#endif
#ifdef USE_GLOP
       GLOP = 8,
#endif
#ifdef USE_CPLEX
       CPLEX = 16,
#endif
#ifdef USE_SCIP
       SCIP = 32,
#endif
     };
     FourierForecaster() = delete;
     FourierForecaster(const std::string& name, enum OptimizationSuite opt_suite);
     ForecasterType GetType() = 0;
     ~FourierForecaster();

     static DATA_REAL_VAL_TYPE infinity() { return std::numeric_limits<DATA_REAL_VAL_TYPE>::infinity(); }
  
     // Whether the given problem type is supported (this will depend on the
     // targets that you linked).
     static bool SupportsOptimizationSuite(FourierForecaster::OptimizationSuite opt_suite);

     // Parses the name of the solver. Returns true if the solver type is
     // successfully parsed as one of the OptimizationProblemType.
     static bool ParseOptimizationSuite(absl::string_view suite,
                                     FourierForecaster::OptimizationSuite* pSuite);

     static bool GetProblemType(OptimizationSuite opt_suite, MPSolver::OptimizationProblemType* pType);

     static bool OptSuiteToString(const OptimizationSuite optimization_suite, std::string & sOptSuite); 

     const std::string& Name() const {
        return name_;  // Set at construction.
     }

     const OptimizationSuite OptSuite() const {
        return opt_suite_;
     }
     
     void set_time_limit(int64 time_limit_milliseconds) {
        DCHECK_GE(time_limit_milliseconds, 0);
        time_limit_ = time_limit_milliseconds;
     }

     // In milliseconds.
     int64 time_limit() const { return time_limit_; }

     // In seconds. Note that this returns a double.
     double time_limit_in_secs() const {
       // static_cast<double> avoids a warning with -Wreal-conversion. This
       // helps catching bugs with unwanted conversions from double to ints.
       return static_cast<double>(time_limit_) / 1000.0;
     }

     // Returns wall_time() in milliseconds since the creation of the solver.
     int64 wall_time() const { return timer_.GetInMs(); }

     const ForecasterStatus get_status() const 
     {
        return (ForecasterStatus) status_;
     }

     DenseDataContainer<DATA_REAL_VAL_TYPE>* get_result()
     {
        return result_.get();
     }

     bool init( );
 
  protected:
     // the name of the forecaster
     //
     const std::string name_;

     // the chosen optimization suite to be used with this Forecaster instance
     //
     enum OptimizationSuite opt_suite_;

     std::unique_ptr<MPSolver> mp_solver_;
      // Time limit in milliseconds (0 = no limit).
      //
     int64 time_limit_;
     WallTimer timer_;
     MPSolver::OptimizationProblemType prob_type_;
     int status_;
     std::unique_ptr<DenseDataContainer<DATA_REAL_VAL_TYPE>> result_;
};

typedef std::pair<DATA_REAL_VAL_TYPE,DATA_REAL_VAL_TYPE> FrequencyNorm;

class FourierForecasterLinear : public FourierForecaster {
   public:
     FourierForecasterLinear(const std::string& name, enum OptimizationSuite opt_suite);
     ForecasterType GetType() override;
     
     // define Prophet-like interface for the classes implementing Forecaster
     //
     //TODO (dpg): finish the method signatures or remove unnecessary methods
     bool fit(const SparseDataContainer<DATA_REAL_VAL_TYPE>& , const DATA_LEN_TYPE&, const DATA_REAL_VAL_TYPE& ) override;
     bool predict( ) override;
     bool predict_trend( ) override;
     bool predict_seasonal_components( ) override;
     bool predicitive_samples( ) override;
     bool validate_inputs( ) override;
     bool validate_column_name( ) override;
     bool setup_dataframe( ) override;
     bool initialize_scales( ) override;     
     bool set_changepoints( ) override;
     bool make_holiday_features( ) override;
     bool make_all_seasonality_features( ) override;
     bool add_regressor( ) override;
     bool add_seasonality( ) override;
     bool add_country_holidays( ) override;
     bool add_group_component( ) override;
     bool parse_seasonality_args( ) override;
     bool sample_posterior_predictive( ) override;
     bool predict_uncertainty( ) override;
     bool sample_model( ) override;
     //
     //END: define Prophet-like interface for the classes implementing Forecaster
     ~FourierForecasterLinear();
   protected:
     void calculate_l1_norm(const std::vector<MPVariable*>& varX, const std::vector<MPVariable*>& varY,
		     const DATA_REAL_VAL_TYPE& lambda, FrequencyNorm& l1_norm); 
};
 
} // ns: forecaster
} // ns: operations_research

#endif // OR_TOOLS_FORECASTER_FOURIER_FORECASTER_H_
