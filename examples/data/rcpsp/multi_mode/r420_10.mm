************************************************************************
file with basedata            : cr420_.bas
initial value random generator: 2088032169
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  123
RESOURCES
  - renewable                 :  4   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       13       12       13
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           9  10  11
   3        3          3           5   6   8
   4        3          3           7  10  14
   5        3          1          12
   6        3          1          12
   7        3          3          11  13  16
   8        3          3           9  11  16
   9        3          2          13  14
  10        3          1          17
  11        3          1          15
  12        3          3          13  16  17
  13        3          1          15
  14        3          2          15  17
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  R 3  R 4  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0    0    0
  2      1     3       9    8    0    2    7    0
         2     7       8    8    2    0    5    0
         3     9       0    8    1    2    5    0
  3      1     1       5    6    0    0    0    9
         2    10       4    0    0    0    3    0
         3    10       0    2    7    7    0    8
  4      1     2       9    9    9    0    0   10
         2     5       7    0    6    0    0   10
         3     7       0    0    5    0    0    9
  5      1     2       0    0    2    0    3    0
         2     3       7    0    0    0    0    5
         3     8       0    0    0    3    3    0
  6      1     3       0    5    0    0    5    0
         2     4       0    4    0    0    1    0
         3     8       0    1    0    7    0    8
  7      1     2       6    8    0    0    0    7
         2     4       4    0    0    7    0    7
         3     9       0    7    8    3    0    3
  8      1     1       9    6    8    0    0    7
         2     6       6    6    0    7    0    6
         3     7       6    6    0    5    5    0
  9      1     1       0    3   10    3    3    0
         2     2       0    2    0    2    3    0
         3     7       7    0    7    0    0    4
 10      1     1       2    0    0    9    5    0
         2     2       0    5    0    8    5    0
         3     3       0    5    0    5    5    0
 11      1     2       0    9    0    0    5    0
         2     8       6    0    0    0    5    0
         3     9       0    6    8    0    0    5
 12      1     5       0    5    0    4    9    0
         2     7       0    0   10    0    8    0
         3    10       0    4    0    3    6    0
 13      1     1       0    8    0    0    0    8
         2     5       8    4    0    0    8    0
         3     5       9    4    0    2    0    8
 14      1     1       8    0    7    1    7    0
         2     6       2    0    0    1    0    9
         3     9       0    0    7    1    0    9
 15      1     3       0    0   10    7    6    0
         2     3       7    1    0    0    0    6
         3     6       6    0    9    7    0    4
 16      1     2       7    0    8    7    0    6
         2     2      10    0    6    0    6    0
         3    10       0    0    0    8    6    0
 17      1     3       0    5    0    0    0    6
         2     5       3    5    0    0    4    0
         3     6       0    4    0    0    1    0
 18      1     0       0    0    0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  R 3  R 4  N 1  N 2
   34   28   20   23   61   71
************************************************************************
