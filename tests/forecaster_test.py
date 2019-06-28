from __future__ import print_function
import numpy as np
import pandas as pd
from scipy.linalg import dft
from ortools.linear_solver import pywraplp

N=100
period=10
S=N/period
parLambda=5
print('lambda=', parLambda)

dataset = [0.0 for j in range(N)]
#dataset[0]=3.+0.j
dataset[25*N/100] = 9.+0.j
#dataset[50]=3.+0.j
dataset[N-25*N/100] = 9.+0.j
idft = np.conj(dft(N))
#time_signal = N*np.fft.ifft(dataset)
time_signal = idft.dot(np.array(dataset))
sampled_time_signal = time_signal[0:N:period]
for i in range(0, S):
    print (sampled_time_signal[i])
solver = pywraplp.Solver('fourier_forecaster',pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
varX = []
varY = []
varS1 = []
varS2 = []
varS3 = []
for i in range(0, N):
    varX.append(solver.NumVar(-solver.infinity(), solver.infinity(), 'x%i' % i))
    varY.append(solver.NumVar(-solver.infinity(), solver.infinity(), 'y%i' % i))
    varS1.append(solver.NumVar(-solver.infinity(), solver.infinity(), 's1%i' % i))
    varS2.append(solver.NumVar(-solver.infinity(), solver.infinity(), 's2%i' % i))
    varS3.append(solver.NumVar(-solver.infinity(), solver.infinity(), 's3%i' % i))

objective = solver.Objective()
for i in range(0, N):
    objective.SetCoefficient(varS1[i],1.0)
    objective.SetCoefficient(varS2[i],parLambda)
    objective.SetCoefficient(varS3[i],parLambda)

objective.SetMinimization()

constrEq = []
for i in range(0, N):
    constrEq.append(solver.Constraint(0.0, 0.0))

for i in range(0, N):
    for j in range(0, N):
        constrEq[i].SetCoefficient(varX[j],np.sin( ( 2.0 * np.pi * i * j ) / N ) )
        constrEq[i].SetCoefficient(varY[j],-np.cos( ( 2.0 * np.pi * i * j ) / N ) )

constrS1_1 = []
for i in range(0, S):
    constrS1_1.append(solver.Constraint(np.real(sampled_time_signal[i]), solver.infinity()))
    constrS1_1[i].SetCoefficient(varS1[i], 1.0)
    for j in range(0, N):
        constrS1_1[i].SetCoefficient(varX[j], np.cos( ( 2.0 * np.pi * i * j ) / N ) )
        constrS1_1[i].SetCoefficient(varY[j], -np.sin( ( 2.0 * np.pi * i * j ) / N ) )

for i in range(S, N):
    constrS1_1.append(solver.Constraint(0.0, solver.infinity()))
    constrS1_1[i].SetCoefficient(varS1[i], 1.0)

constrS1_2 = []
for i in range(0, S):
    constrS1_2.append(solver.Constraint(-np.real(sampled_time_signal[i]), solver.infinity()))
    constrS1_2[i].SetCoefficient(varS1[i], 1.0)
    for j in range(0, N):
        constrS1_2[i].SetCoefficient(varX[j], -np.cos( ( 2.0 * np.pi * i * j ) / N ) )
        constrS1_2[i].SetCoefficient(varY[j], np.sin( ( 2.0 * np.pi * i * j ) / N ) )

constrS2_1 = []
for i in range(0, N):
    constrS2_1.append(solver.Constraint(0.0, solver.infinity()))
    constrS2_1[i].SetCoefficient(varX[i], -1.0 )
    constrS2_1[i].SetCoefficient(varS2[i], 1.0 )

constrS2_2 = []
for i in range(0, N):
    constrS2_2.append(solver.Constraint(0.0, solver.infinity()))
    constrS2_2[i].SetCoefficient(varX[i], 1.0 )
    constrS2_2[i].SetCoefficient(varS2[i], 1.0 )

constrS3_1 = []
for i in range(0, N):
    constrS3_1.append(solver.Constraint(0.0, solver.infinity()))
    constrS3_1[i].SetCoefficient(varY[i], -1.0 )
    constrS3_1[i].SetCoefficient(varS3[i], 1.0 )

constrS3_2 = []
for i in range(0, N):
    constrS3_2.append(solver.Constraint(0.0, solver.infinity()))
    constrS3_2[i].SetCoefficient(varY[i], 1.0 )
    constrS3_2[i].SetCoefficient(varS3[i], 1.0 )

print('Number of variables =', solver.NumVariables())
print('Number of constraints =', solver.NumConstraints())

# Solve the system.
status = solver.Solve()
# Check that the problem has an optimal solution.
if status != pywraplp.Solver.OPTIMAL:
    print("The problem does not have an optimal solution!")
    exit(1)

print('Solution:')
y = []
for i in range(0, N):
    y.append(complex(varX[i].solution_value(), varY[i].solution_value()))
    print('y_{re}[', i, ']=', varX[i].solution_value(), 'y_{im}[', i, ']=', varY[i].solution_value())
#idft = np.conj(dft(N))
recovered_time_signal = idft.dot(np.array(y))
for i in range(0, N):
    print('x_{re}[', i, ']=', np.real(recovered_time_signal[i]), 'x_{im}[', i, ']=', np.imag(recovered_time_signal[i]))
