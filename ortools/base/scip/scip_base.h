#ifndef OR_TOOLS_BASE_SCIP_SCIP_BASE_H_
#define OR_TOOLS_BASE_SCIP_SCIP_BASE_H_

#if defined(USE_SCIP)

// Heuristics
// Our own version of SCIP_CALL to do error management.
// TODO(user): The error management could be improved, especially
// for the Solve method. We should return an error status (did the
// solver encounter problems?) and let the user query the result
// status (optimal, infeasible, ...) with a separate method. This is a
// common API for solvers. The API change in all existing code might
// not be worth it.
#define ORTOOLS_SCIP_CALL(x) CHECK_EQ(SCIP_OKAY, x)



#endif  // defined(USE_SCIP)

#endif  // OR_TOOLS_BASE_SCIP_SCIP_BASE_H_