#include "ortools/forecaster/scip_forecaster.h"

#if defined(USE_SCIP)

namespace operations_research {
namespace forecaster {
    
void SCIPForecaster::Reset() {
  //TO DO: finish this method , look at SCIPInterface::Reset()
  // -- dimitarpg13 6-31-19 --
}

void SCIPForecaster::CreateSCIP() {
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


void SCIPForecaster::DeleteSCIP() {
  CHECK(scip_ != nullptr);
  ORTOOLS_SCIP_CALL(SCIPreleaseVar(scip_, &objective_offset_variable_));
  for (int i = 0; i < scip_variables_.size(); ++i) {
    ORTOOLS_SCIP_CALL(SCIPreleaseVar(scip_, &scip_variables_[i]));
  }
  scip_variables_.clear();
  for (int j = 0; j < scip_constraints_.size(); ++j) {
    ORTOOLS_SCIP_CALL(SCIPreleaseCons(scip_, &scip_constraints_[j]));
  }
  scip_constraints_.clear();
  ORTOOLS_SCIP_CALL(SCIPfree(&scip_));
  scip_ = nullptr;
}

void SCIPForecaster::InvalidateSolutionSynchronization() {
  if (sync_status_ == SOLUTION_SYNCHRONIZED) {
    sync_status_ = MODEL_SYNCHRONIZED;
  }
}

// Not cached.
void SCIPForecaster::SetOptimizationDirection(bool maximize) {
  InvalidateSolutionSynchronization();
  ORTOOLS_SCIP_CALL(SCIPfreeTransform(scip_));
  ORTOOLS_SCIP_CALL(SCIPsetObjsense(
      scip_, maximize ? SCIP_OBJSENSE_MAXIMIZE : SCIP_OBJSENSE_MINIMIZE));
}

void SCIPForecaster::SetVariableBounds(int var_index, double lb, double ub) {
  InvalidateSolutionSynchronization();
  if (variable_is_extracted(var_index)) {
    // Not cached if the variable has been extracted.
    DCHECK_LT(var_index, last_variable_index_);
    ORTOOLS_SCIP_CALL(SCIPfreeTransform(scip_));
    ORTOOLS_SCIP_CALL(SCIPchgVarLb(scip_, scip_variables_[var_index], lb));
    ORTOOLS_SCIP_CALL(SCIPchgVarUb(scip_, scip_variables_[var_index], ub));
  } else {
    sync_status_ = MUST_RELOAD;
  }
}


}  // namespace: forecaster

}  // namespace: operations_research

#endif // defined(USE_SCIP)