g++ -fPIC -std=c++11 -fwrapv -O0 -g -I. -Iortools/gen -I/opt/or-tools/dependencies/install/include -I/opt/or-tools/dependencies/install/include -I/opt/or-tools/dependencies/install/include -I/opt/or-tools/dependencies/install/include -I/opt/or-tools/dependencies/install/include/coin -I/opt/or-tools/dependencies/install/include -I/opt/or-tools/dependencies/install/include/coin -I/opt/or-tools/dependencies/install/include -I/opt/or-tools/dependencies/install/include/coin -DUSE_CLP -I/opt/or-tools/dependencies/install/include -I/opt/or-tools/dependencies/install/include/coin -I/opt/or-tools/dependencies/install/include -I/opt/or-tools/dependencies/install/include/coin -DUSE_CBC -DUSE_GUROBI -Wno-deprecated -DUSE_GLOP -DUSE_BOP  -DNO_CONFIG_HEADER -I/opt/gurobi800/linux64/include -DUSE_GUROBI -I/opt/CPLEX_Studio_Community128/cplex/include -DUSE_CPLEX \
	 objs/pwl_mip_solver_example.o \
	   -L/opt/or-tools/dependencies/install/lib -lgflags -L/opt/or-tools/dependencies/install/lib -lglog -L/opt/or-tools/dependencies/install/lib/ -lprotobuf -L/opt/or-tools/dependencies/install/lib -lCbcSolver -lCbc -lOsiCbc -L/opt/or-tools/dependencies/install/lib -lCgl -L/opt/or-tools/dependencies/install/lib -lClpSolver -lClp -lOsiClp -L/opt/or-tools/dependencies/install/lib -lOsi -L/opt/or-tools/dependencies/install/lib -lCoinUtils -L/opt/or-tools/lib -lortools -lz -lrt -lpthread -Wl,-rpath,'$ORIGIN' -Wl,-rpath,'$ORIGIN/../lib' -Wl,-rpath,'$ORIGIN/../dependencies/install/lib64' -Wl,-rpath,'$ORIGIN/../dependencies/install/lib' \
	    -o bin/pwl_mip_solver_example
