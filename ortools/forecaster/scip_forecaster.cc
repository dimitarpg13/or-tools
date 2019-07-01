#include "ortools/forecaster/scip_forecaster.h"

#if defined(USE_SCIP)

namespace operations_research {
namespace forecaster {
    
void ScipForecaster::Reset() {

}

void ScipForecaster::CreateSCIP() {
  ORTOOLS_SCIP_CALL(SCIPcreate(&scip_));
  ORTOOLS_SCIP_CALL(SCIPincludeDefaultPlugins(scip_));
  // Set the emphasis to enum SCIP_PARAMEMPHASIS_FEASIBILITY. Do not print
  // the new parameter (quiet = true).
  if (FLAGS_scip_feasibility_emphasis) {
    ORTOOLS_SCIP_CALL(SCIPsetEmphasis(scip_, SCIP_PARAMEMPHASIS_FEASIBILITY,
                                      /*quiet=*/true));
  }
  // Default clock type. We use wall clock time because getting CPU user seconds
  // involves calling times() which is very expensive.
  ORTOOLS_SCIP_CALL(
      SCIPsetIntParam(scip_, "timing/clocktype", SCIP_CLOCKTYPE_WALL));
  ORTOOLS_SCIP_CALL(SCIPcreateProb(scip_, name_.c_str(), nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr));
  ORTOOLS_SCIP_CALL(SCIPsetObjsense(
      scip_, maximize_ ? SCIP_OBJSENSE_MAXIMIZE : SCIP_OBJSENSE_MINIMIZE));
  // SCIPaddObjoffset cannot be used at the problem building stage. So we handle
  // the objective offset by creating a dummy variable.
  objective_offset_variable_ = nullptr;
  // The true objective coefficient will be set in ExtractObjective.
  double dummy_obj_coef = 0.0;
  ORTOOLS_SCIP_CALL(SCIPcreateVar(scip_, &objective_offset_variable_, "dummy",
                                  1.0, 1.0, dummy_obj_coef,
                                  SCIP_VARTYPE_CONTINUOUS, true, false, nullptr,
                                  nullptr, nullptr, nullptr, nullptr));
  ORTOOLS_SCIP_CALL(SCIPaddVar(scip_, objective_offset_variable_));
}
}  // namespace: forecaster

}  // namespace: operations_research

#endif // defined(USE_SCIP)