#include <cmath>
#include <math.h>
#include "ortools/forecaster/fourier_forecaster.h"

namespace operations_research {
namespace forecaster {

//TODO (dpg):
FourierForecaster::FourierForecaster(const std::string& name, enum OptimizationSuite opt_suite) : 
       name_(name), opt_suite_(opt_suite), status_(SUCCESS)	
{
   bool res = GetProblemType(opt_suite, &prob_type_);
   if (!res) {
      status_ |= UNKNOWN_PROBLEM_TYPE; 
   }
}

//TODO (dpg):
Forecaster::ForecasterType FourierForecasterLinear::GetType() {
   return Forecaster::ForecasterType::FourierLinear; 
}

bool FourierForecaster::init( ) {
    timer_.Restart();
    mp_solver_ = std::unique_ptr<MPSolver>(new MPSolver(name_, prob_type_));
    if (!(status_ &= UNKNOWN_PROBLEM_TYPE))
       return true;
    else
       return false;
}

//TODO (dpg):
FourierForecaster::~FourierForecaster() {
}


// static FourierForecaster methods
//

bool FourierForecaster::GetProblemType(FourierForecaster::OptimizationSuite opt_suite, MPSolver::OptimizationProblemType* pType) {
#ifdef USE_GUROBI
    if (opt_suite == FourierForecaster::GUROBI) { 
        *pType = MPSolver::GUROBI_LINEAR_PROGRAMMING;
        return true;
    }
#endif
#ifdef USE_GLPK
    else if (opt_suite == FourierForecaster::GLPK) { 
        *pType = MPSolver::GLPK_LINEAR_PROGRAMMING;
        return true;
    }
#endif
#ifdef USE_CLP
    else if (opt_suite == FourierForecaster::CLP) { 
        *pType = MPSolver::CLP_LINEAR_PROGRAMMING;
        return true;
    }
#endif
#ifdef USE_GLOP
    else if (opt_suite == FourierForecaster::GLOP) { 
        *pType = MPSolver::GLOP_LINEAR_PROGRAMMING;
        return true;
    }
#endif
#ifdef USE_CPLEX
    else if (opt_suite == FourierForecaster::CPLEX) { 
        *pType = MPSolver::CPLEX_LINEAR_PROGRAMMING;
        return true;
    }
#endif
#ifdef USE_SCIP
    else if (opt_suite == FourierForecaster::SCIP) {
        *pType = MPSolver::SCIP_MIXED_INTEGER_PROGRAMMING;
	return true;
    }
#endif
    else return false;
}


bool FourierForecaster::SupportsOptimizationSuite(FourierForecaster::OptimizationSuite opt_suite) {
#ifdef USE_GUROBI
    if (opt_suite == GUROBI) return true;
#endif
#ifdef USE_GLPK
    if (opt_suite == GLPK) return true;
#endif
#ifdef USE_CLP
    if (opt_suite == CLP) return true;
#endif
#ifdef USE_GLOP
    if (opt_suite == GLOP) return true;
#endif
#ifdef USE_CPLEX
    if (opt_suite == CPLEX) return true;
#endif
    return false;
}


bool FourierForecaster::ParseOptimizationSuite(absl::string_view suite,
                                       FourierForecaster::OptimizationSuite* pSuite) {
#if defined(USE_GUROBI)
   if (suite == "gurobi") {
     *pSuite = FourierForecaster::GUROBI;
#endif
#if defined(USE_GLPK)
   } else if (suite == "glpk") {
    *pSuite = FourierForecaster::GLPK;
#endif
#if defined(USE_CLP)
  } else if (suite == "clp") {
    *pSuite = FourierForecaster::CLP;
#endif
#if defined(USE_GLOP)
  } else if (suite == "glop") {
    *pSuite = FourierForecaster::GLOP;
#endif
#if defined(USE_CPLEX)
  } else if (suite == "cplex") {
    *pSuite = FourierForecaster::CPLEX;
#endif
  } else {
    return false;
  }
  return true;
}

bool FourierForecaster::OptSuiteToString(const OptimizationSuite optimization_suite, std::string & sOptSuite) {
  bool res;
  switch (optimization_suite) {
#ifdef USE_GUROBI
    case GUROBI:
      sOptSuite = "GUROBI";
      res = true;
      break;
#endif
#ifdef USE_GLPK
    case GLPK:
      sOptSuite = "GLPK";
      res = true;
      break;
#endif
#ifdef USE_CLP
    case CLP:
      sOptSuite = "CLP";
      res = true;
      break;
#endif
#ifdef USE_GLOP
    case GLOP:
      sOptSuite = "GLOP";
      res = true;
      break;
#endif
#ifdef USE_CPLEX
    case CPLEX:
      sOptSuite = "CPLEX";
      res = true;
      break;
#endif
    default:
      sOptSuite = "Unknown";
      res = false;
      break;
  }
  return res;
}


FourierForecasterLinear::FourierForecasterLinear(const std::string& name, enum OptimizationSuite opt_suite) :
   FourierForecaster(name, opt_suite)  
{
//TODO: (dpg) finish this
}

FourierForecasterLinear::~FourierForecasterLinear() {

}

// Prophet-like interface for all classes implementing Forecaster
// TODO (dpg): finish/remove the methods signatures
// LP Problem solved by the Fourier forecaster
//


bool FourierForecasterLinear::fit(const SparseDataContainer<DATA_REAL_VAL_TYPE>& data, const DATA_LEN_TYPE& N, const DATA_REAL_VAL_TYPE& lambda ) { 
   if (!init()) 
        return false;

   bool result = false;

   const auto S = data.size();
   const auto d = N;

   // initialize all LP variables including the slack ones
   std::vector<MPVariable*> varX(d,nullptr);
   std::vector<MPVariable*> varY(d,nullptr);
   for (int i = 0; i < d; ++i) {
      std::string strX = "x" + std::to_string(i); 
      varX[i] = mp_solver_->MakeNumVar(-infinity(), infinity(), strX.c_str());
      std::string strY = "y" + std::to_string(i); 
      varY[i] = mp_solver_->MakeNumVar(-infinity(), infinity(), strY.c_str());
   }
   std::vector<MPVariable*> varS1(d,nullptr);
   for (int i = 0; i < d; ++i) {
      std::string strS1 = "s1" + std::to_string(i);
      varS1[i] = mp_solver_->MakeNumVar(-infinity(), infinity(), strS1.c_str());
   } 
   std::vector<MPVariable*> varS2(d,nullptr);
   for (int i = 0; i < d; ++i) {
      std::string strS2 = "s2" + std::to_string(i);
      varS2[i] = mp_solver_->MakeNumVar(-infinity(), infinity(), strS2.c_str());
   } 
   std::vector<MPVariable*> varS3(d,nullptr);
   for (int i = 0; i < d; ++i) {
      std::string strS3 = "s3" + std::to_string(i);
      varS3[i] = mp_solver_->MakeNumVar(-infinity(), infinity(), strS3.c_str());
   } 
  
   // create the objective function 
   MPObjective* const objective = mp_solver_->MutableObjective();
   for (int i = 0; i < d; ++i) { 
      objective->SetCoefficient(varS1[i], 1);
   }
   for (int i = 0; i < d; ++i) { 
      objective->SetCoefficient(varS2[i], lambda);
   }
   for (int i = 0; i < d; ++i) { 
      objective->SetCoefficient(varS3[i], lambda);
   }
   objective->SetMinimization();

   // add N_ equality constraints
   std::vector<MPConstraint*> constrEq(N,nullptr);
   // Note: in our forecaster problem we are going to use real valued signals in time domain
   // which implies that the DFT of the signal will be an array of complex numbers where
   // the first half of the array is complex conjugate of the second half. If the array is of 
   // length N and the frequency domain array values are denoted by y_r[k] + i* y_i[k], 0 <= k <= N/2
   // then the inverse DFT x[l], 0 <= l <= N-1 is given with:
   //
   // x[l] = \sum_{k=0}^{\frac{N}{2}} y_r[k] {\cos}(\frac{2{\pi}kl}{N}) - y_i[l] {\sin}(\frac{2{\pi}kl}{N})
   //
   for (int i = 0; i < N; ++i) {
         constrEq[i] = mp_solver_->MakeRowConstraint(0.0, 0.0);
      for (int j = 0; j < d; ++j) {
         constrEq[i]->SetCoefficient(varX[j], sin( ( 2.0 * M_PI * i * j ) / N ) );
         constrEq[i]->SetCoefficient(varY[j], -cos( ( 2.0 * M_PI * i * j ) / N ) );
      }
   }

   // add the first constraint set for the first set of slack variables s1:
   //
   // s^{1} - P_{\Omega}( x - F^{*}_{re} + F^{*}_{im} ) >= 0
   std::vector<MPConstraint*> constrS1_1(d,nullptr);
   for (int i = 0; i < S; ++i) {
      constrS1_1[i] = mp_solver_->MakeRowConstraint(data[i].second, infinity());
      constrS1_1[i]->SetCoefficient(varS1[i], 1.0 );
      for (int j = 0; j < d; ++j) {
         constrS1_1[i]->SetCoefficient(varX[j], cos( ( 2.0 * M_PI * i * j ) / N ) );
         constrS1_1[i]->SetCoefficient(varY[j], -sin( ( 2.0 * M_PI * i * j ) / N ) );
      }
   }

   for (int i = S; i < d; ++i) {
      constrS1_1[i] = mp_solver_->MakeRowConstraint(0, infinity());
      constrS1_1[i]->SetCoefficient(varS1[i], 1.0 );
   }

   // add the second constraint set for the first set of slack variables s1:
   //
   // s^{1} + P_{\Omega}( x - F^{*}_{re} + F^{*}_{im} ) >= 0
   std::vector<MPConstraint*> constrS1_2(S,nullptr);
   for (int i = 0; i < S; ++i) {
      constrS1_2[i] = mp_solver_->MakeRowConstraint(-data[i].second, infinity());
      constrS1_2[i]->SetCoefficient(varS1[i], 1.0 );
      for (int j = 0; j < d; ++j) {
         constrS1_2[i]->SetCoefficient(varX[j], -cos( ( 2.0 * M_PI * i * j ) / N ) );
         constrS1_2[i]->SetCoefficient(varY[j], sin( ( 2.0 * M_PI * i * j ) / N ) );
      }
   }
   
   // add the first constraint set for the second set of slack variables s2:
   //
   // s^{2} - y_{re} >= 0
   std::vector<MPConstraint*> constrS2_1(d,nullptr);
   for (int i = 0; i < d; ++i) {
      constrS2_1[i] = mp_solver_->MakeRowConstraint(0, infinity());
      constrS2_1[i]->SetCoefficient(varX[i], -1.0 );
      constrS2_1[i]->SetCoefficient(varS2[i], 1.0 );
   }

   // add the second constraint set for the second set of slack variables s2:
   //
   // s^{2} + y_{re} >= 0
   std::vector<MPConstraint*> constrS2_2(d,nullptr);
   for (int i = 0; i < d; ++i) {
      constrS2_2[i] = mp_solver_->MakeRowConstraint(0, infinity());
      constrS2_2[i]->SetCoefficient(varX[i], 1.0 );
      constrS2_2[i]->SetCoefficient(varS2[i], 1.0 );
   }

   // add the first constraint set for the third set of slack variables s3:
   //
   // s^{3} - y_{im} >= 0
   std::vector<MPConstraint*> constrS3_1(d,nullptr);
   for (int i = 0; i < d; ++i) {
      constrS3_1[i] = mp_solver_->MakeRowConstraint(0, infinity());
      constrS3_1[i]->SetCoefficient(varY[i], -1.0 );
      constrS3_1[i]->SetCoefficient(varS3[i], 1.0 );
   }

   // add the second constraint set for the third set of slack variables s3:
   //
   // s^{3} + y_{im} >= 0
   std::vector<MPConstraint*> constrS3_2(d,nullptr);
   for (int i = 0; i < d; ++i) {
      constrS3_2[i] = mp_solver_->MakeRowConstraint(0, infinity());
      constrS3_2[i]->SetCoefficient(varY[i], 1.0 );
      constrS3_2[i]->SetCoefficient(varS3[i], 1.0 );
   }

   LOG(INFO) << "Number of variables = " << mp_solver_->NumVariables();
   LOG(INFO) << "Number of constraints = " << mp_solver_->NumConstraints();

   const MPSolver::ResultStatus result_status = mp_solver_->Solve();
   // Check that the problem has an optimal solution.
   if (result_status != MPSolver::OPTIMAL) {
      LOG(FATAL) << "The problem does not have an optimal solution!";
   } else {

      result = true;
   }

   result_ = std::unique_ptr<DenseDataContainer<DATA_REAL_VAL_TYPE>>(new DenseDataContainer<DATA_REAL_VAL_TYPE>(d));
   for (int i=0; i < d; ++i) {
      for (int j=0; j < d; ++j) {
         (*result_)[i] += varX[j]->solution_value() * cos( ( 2.0 * M_PI * i * j ) / N ) - varY[j]->solution_value() * sin ( ( 2.0 * M_PI * i * j ) / N ) ; 
      }
   }
   double sumOfAbsYRe=0.0, sumOfAbsYIm=0.0;
   for (int i = 0; i < N; ++i) {
     LOG(INFO) << "varX[" << i << "]=" << (*varX[i]).solution_value() << ", " << "varY[" << i << "]=" << (*varY[i]).solution_value() << std::endl;
     sumOfAbsYRe += std::fabs((*varX[i]).solution_value());
     sumOfAbsYIm += std::fabs((*varY[i]).solution_value());
   }
   LOG(INFO) << "sumOfAbsYRe = " << sumOfAbsYRe << ", " << "sumOfAbsYIm = " << sumOfAbsYIm;


   double errAbs=0.0, errPerc=0.0, sumOfAbs=0.0;
   for (int i = 0; i < S; ++i) {
      errAbs += fabs((*result_)[i] - data[i].second);
      sumOfAbs += fabs(data[i].second); 
   }
   LOG(INFO) << "Absolute error: " << errAbs << ", Percent error = " << errAbs/sumOfAbs*100.0; 

   return result;
}

bool FourierForecasterLinear::predict( ) {
   return false;
}

bool FourierForecasterLinear::predict_trend( ) {
   return false;
}

bool FourierForecasterLinear::predict_seasonal_components( ) {
   return false;
}

bool FourierForecasterLinear::predicitive_samples( ) {
   return false;
}

bool FourierForecasterLinear::validate_inputs( ) { 
   return false;
}

bool FourierForecasterLinear::validate_column_name( ) {
   return false;
}

bool FourierForecasterLinear::setup_dataframe( ) {
   return false;
}

bool FourierForecasterLinear::initialize_scales( ) {
   return false;
}
     
bool FourierForecasterLinear::set_changepoints( ) {
   return false;
}

bool FourierForecasterLinear::make_holiday_features( ) {
   return false;
}

bool FourierForecasterLinear::make_all_seasonality_features( ) {
   return false;
}

bool FourierForecasterLinear::add_regressor( ) {
   return false;
}

bool FourierForecasterLinear::add_seasonality( ) {
   return false;
}

bool FourierForecasterLinear::add_country_holidays( ) {
   return false;
}

bool FourierForecasterLinear::add_group_component( ) {
   return false;
}

bool FourierForecasterLinear::parse_seasonality_args( ) {
   return false;
}

bool FourierForecasterLinear::sample_posterior_predictive( ) {
   return false;
}

bool FourierForecasterLinear::predict_uncertainty( ) {
   return false;
}

bool FourierForecasterLinear::sample_model( ) {
   return false;
}
//
//END: Prophet-like interface for all classes implementing Forecaster

} // ns: forecaster

} // ns: operations_research   
