************************************************************************
file with basedata            : cm119_.bas
initial value random generator: 1890763185
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  106
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       36        9       36
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        1          2           5   7
   3        1          3          10  11  17
   4        1          3           6   8  12
   5        1          3           9  11  13
   6        1          2          10  14
   7        1          2           8   9
   8        1          2          10  13
   9        1          3          15  16  17
  10        1          1          15
  11        1          1          16
  12        1          1          14
  13        1          2          14  16
  14        1          2          15  17
  15        1          1          18
  16        1          1          18
  17        1          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       0    5    0    3
  3      1    10       0    6    7    0
  4      1    10       0    6    7    0
  5      1     9       0    7    0    2
  6      1     7       1    0    0    7
  7      1     1       6    0   10    0
  8      1    10       1    0    3    0
  9      1    10       3    0    0    4
 10      1     3       0    5    4    0
 11      1     8       5    0    6    0
 12      1     9       3    0    5    0
 13      1     7       0    4    4    0
 14      1     2       3    0    0    1
 15      1     2       1    0    0    9
 16      1     9       7    0    4    0
 17      1     6       0    5    0    4
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   12   16   50   30
************************************************************************