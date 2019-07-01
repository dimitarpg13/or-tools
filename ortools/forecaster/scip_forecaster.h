#ifndef OR_TOOLS_FORECASTER_SCIP_FORECASTER_H_
#define OR_TOOLS_FORECASTER_SCIP_FORECASTER_H_

#if defined(USE_SCIP)

#include <vector>
#include <string>
#include <utility>
#include <math.h>
#include <memory>
#include <unordered_map>
#include <complex.h>
#include <type_traits>
#include <limits>
#include "ortools/base/timer.h"
#include "ortools/base/scip/scip_base.h"
//#include "ortools/linear_solver/linear_solver.h"
//#include "ortools/base/commandlineflags.h"
//#include "ortools/base/hash.h"
//#include "ortools/base/integral_types.h"
#include "ortools/base/logging.h"
//#include "ortools/base/port.h"
#include "ortools/base/stringprintf.h"
#include "ortools/base/timer.h"
#include "forecaster.h"
#include "scip/scip.h"
#include "scip/scipdefplugins.h"

DECLARE_bool(scip_feasibility_emphasis);

namespace operations_research {
namespace forecaster {

class SCIPForecaster : public Forecaster {

   //TO DO (dimitarpg13): most likely such synchronization will not be necessary 
   // for the SCIP-based forecaster so remove it.
   enum SynchronizationStatus {
    // The underlying solver (CLP, GLPK, ...) and MPSolver are not in
    // sync for the model nor for the solution.
    MUST_RELOAD,
    // The underlying solver and MPSolver are in sync for the model
    // but not for the solution: the model has changed since the
    // solution was computed last.
    MODEL_SYNCHRONIZED,
    // The underlying solver and MPSolver are in sync for the model and
    // the solution.
    SOLUTION_SYNCHRONIZED
  };
protected:
  // Change synchronization status from SOLUTION_SYNCHRONIZED to
  // MODEL_SYNCHRONIZED. To be used for model changes.
  void InvalidateSolutionSynchronization();
  void Reset();
  void SetOptimizationDirection(bool maximize); 
  void CreateSCIP();
  void DeleteSCIP();
  void SetVariableBounds(int var_index, double lb, double ub);

  //TO DO (dimitarpg13): do we really need this state variable?
  bool variable_is_extracted(int var_index) const {
    return variable_is_extracted_[var_index];
  }
  // Indicates whether the model and the solution are synchronized.
  SynchronizationStatus sync_status_;

   // The name of the linear programming problem.
  const std::string name_;
 // Optimization direction.
  bool maximize_;

  // Index in MPSolver::constraints_ of last constraint extracted.
  int last_constraint_index_;

  // Index in MPSolver::sos_constraints_
  int last_sos_constraint_index_;

  // Index in MPSolver::variables_ of last variable extracted.
  int last_variable_index_;

  // The value of the objective function.
  double objective_value_;
  SCIP* scip_;
  SCIP_VAR* objective_offset_variable_;
  std::vector<SCIP_VAR*> scip_variables_;
  std::vector<SCIP_CONS*> scip_constraints_;

   // Whether variables have been extracted to the underlying interface.
  std::vector<bool> variable_is_extracted_;


};

} // namespace operations_research

} // namespace forecaster

#endif // defined(USE_SCIP)

#endif // OR_TOOLS_FORECASTER_CBC_FORECASTER_H_
