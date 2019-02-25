"""programming examples that show how to use the PWL solver."""

from ortools.linear_solver import pywraplp

def RunPWLExampleWithScalarXDomainAndTwoContinuousVars(optimization_suite):
    """
    Example of a PWL three-valued function defined in scalar X domain.
    Two continuous variables are defined.
    """
    problemName = 'scalar x domain and two continuous variables'
    solver = pywraplp.PwlSolver(problemName, optimization_suite)

    x = [ [ 1.0, 2.0, 4.0 ] ]
    y = [ 1.0, 1.5, 2.0 ]
    A = [ [ 2.0 ], [ 1.0 ] ]
    B = [ [ -1.0, 2.8 ], [ 2.8, 1.0 ] ]
    b = [ 1.0, 2.0 ]
    c = [ 0.5 ]
    d = [ 10.8, 13.8 ]

    SetProblemParameters(solver,x,y,A,B,b,c,d)
    PrintProblemHeader(solver,problemName,optimization_suite)
    SolveAndPrintSolution(solver)

def RunPWLExampleWithScalarXDomainAndThreeContinuousVars(optimization_suite):
    """
    Example of a PWL three-valued function defined in scalar X domain.
    Three continuous variables are defined.
    """
    problemName = 'scalar x domain and three continuous variables'
    solver = pywraplp.PwlSolver(problemName, optimization_suite)

    x = [ [ 1.0, 2.0, 4.0 ] ]
    y = [ 1.0, 1.5, 2.0 ]
    A = [ [ 1.0 ], [ 1.0 ], [ 1.0 ] ];
    B = [ [ 1.0, 1.0, 0.0 ], [ 1.0, 0.0, 1.0 ], [ 0.0, 1.0, 1.0 ] ]
    b = [ 1.0, 0.25, 0.25 ]
    c = [ 0.5 ]
    d = [ 10.0, 10.0, 10.0 ]

    SetProblemParameters(solver,x,y,A,B,b,c,d)
    PrintProblemHeader(solver,problemName,optimization_suite)
    SolveAndPrintSolution(solver)

def RunPWLExampleWith2DimXDomainAndTwoContinuousVars(optimization_suite):
    """
    Example of a PWL three-valued function defined in 2-dimensional X domain.
    Two continuous variables are defined.
    """
    problemName = 'two-dimensional x domain and two continuous variables'
    solver = pywraplp.PwlSolver(problemName, optimization_suite)

    x = [ [ 1.0, 2.0, 4.0 ], [ 1.0, 2.0, 4.0 ] ]
    y = [ 1.0, 1.5, 2.0 ]
    A = [ [ 2.0, 1.0 ], [ 1.0, 0.5 ] ]
    B = [ [ -1.0, 2.8 ], [ 2.8, 1.0 ] ]
    b = [ 1.0, 2.0 ]
    c = [ 0.5, 0.25 ]
    d = [ 10.8, 13.8 ]

    SetProblemParameters(solver,x,y,A,B,b,c,d)
    PrintProblemHeader(solver,problemName,optimization_suite)
    SolveAndPrintSolution(solver)

def RunPWLExampleWith2DimXDomainAndThreeContinuousVars(optimization_suite):
    """
    Example of a PWL three-valued function defined in scalar X domain.
    Three continuous variables are defined.
    """
    problemName = 'two-dimensional x domain and three continuous variables'
    solver = pywraplp.PwlSolver(problemName, optimization_suite)

    x = [ [ 1.0, 2.0, 4.0 ], [ 1.0, 2.0, 4.0 ] ]
    y = [ 1.0, 1.5, 2.0 ]
    A = [ [ 1.0, 0.5 ], [ 1.0, 0.5 ], [ 1.0, 0.5 ] ];
    B = [ [ 1.0, 1.0, 0.0 ], [ 1.0, 0.0, 1.0 ], [ 0.0, 1.0, 1.0 ] ]
    b = [ 1.0, 0.25, 0.25 ]
    c = [ 0.5, 0.25 ]
    d = [ 10.0, 10.0, 10.0 ]

    SetProblemParameters(solver,x,y,A,B,b,c,d)
    PrintProblemHeader(solver,problemName,optimization_suite)
    SolveAndPrintSolution(solver)

def SolveAndPrintSolution(solver):
    """Solve the PWL problem and print the solution."""
    print(('Total number of integer and continuous variables = %d' % solver.NumVariables()))
    print(('Number of continuous variables = %d' % solver.NumRealVariables()))

    result_status = solver.Solve()

    # The problem has an optimal solution.
    assert result_status == pywraplp.Solver.OPTIMAL

    # The solution looks legit (when using solvers others than
    # GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
    assert solver.VerifySolution(1e-7, True)

    print(('Problem solved in %f milliseconds' % solver.wall_time()))

    # The objective value of the solution.
    print(('Optimal objective value = %f' % solver.Objective().Value()))

    numIntVars = solver.NumVariables() - solver.NumRealVariables()
    numContVars = solver.NumRealVariables()
    print('Solution values for the integer variables:')
    for i in range(numIntVars):
        variable = solver.GetVariable(i)
        print(('lambda_%d = %d' % ((i+1), variable.solution_value())))

    print('Solution values for the continuous variables:')
    for j in range(numContVars):
        variable = solver.GetVariable(numIntVars+j)
        print(('z_%d = %f' % ((j+1), variable.solution_value())))

    print('Advanced usage:')
    print(('Problem solved in %d branch-and-bound nodes' % solver.nodes()))

def PrintProblemHeader(solver, name, optimization_suite):
    print()
    title = ('Mixed integer programming example with PWL function in %s' % name)
    titleLen = len(title)
    print('-' * titleLen)
    print(title)
    print('-' * titleLen)
    print(('Solver type: %s' % optimization_suite))
    print(('Total number of variables = %d' % solver.NumVariables()))
    print(('Number of continuous variables = %d' % solver.NumRealVariables()))
    print(('Number of integer variables = %d' % (solver.NumVariables() - solver.NumRealVariables())))
    print(('Number of discrete points = %d' % solver.NumOfXPoints()))
    print(('Dimension of X domain = %d' % solver.DimOfXPoint()))
    print(('Number of constraints = %d' % solver.NumConstraints()))

def SetProblemParameters(solver, x, y, A, B, b, c, d):
    solver.SetXValues(x)
    solver.SetYValues(y)
    solver.SetParameter(b, pywraplp.PwlSolver.bVector)
    solver.SetParameter(c, pywraplp.PwlSolver.cVector)
    solver.SetParameter(d, pywraplp.PwlSolver.dVector)
    solver.SetParameter(A, pywraplp.PwlSolver.AMatrix)
    solver.SetParameter(B, pywraplp.PwlSolver.BMatrix)

def main():
    RunPWLExampleWithScalarXDomainAndTwoContinuousVars(pywraplp.PwlSolver.GUROBI)
    RunPWLExampleWithScalarXDomainAndThreeContinuousVars(pywraplp.PwlSolver.GUROBI)
    RunPWLExampleWith2DimXDomainAndTwoContinuousVars(pywraplp.PwlSolver.GUROBI)
    RunPWLExampleWith2DimXDomainAndThreeContinuousVars(pywraplp.PwlSolver.GUROBI)

if __name__ == '__main__':
    main()
