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

class ScipForecaster : public Forecaster {

protected:
  void Reset();
  void CreateSCIP();
  void DeleteSCIP();
  
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
    

};

} // namespace operations_research

} // namespace forecaster

#endif // defined(USE_SCIP)

#endif // OR_TOOLS_FORECASTER_CBC_FORECASTER_H_
