************************************************************************
file with basedata            : cr144_.bas
initial value random generator: 1420
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  126
RESOURCES
  - renewable                 :  1   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       22       15       22
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2          10  14
   3        3          3           5   7   9
   4        3          2           6   8
   5        3          3           6   8  11
   6        3          1          17
   7        3          2          12  15
   8        3          2          13  14
   9        3          3          10  12  15
  10        3          1          16
  11        3          2          12  14
  12        3          1          13
  13        3          2          16  17
  14        3          3          15  16  17
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0
  2      1     3       0    6    6
         2     7       0    5    5
         3    10       0    2    5
  3      1     1       0    4    8
         2     6       7    3    8
         3    10       0    3    8
  4      1     1       0    7    6
         2     2       4    5    5
         3    10       0    4    2
  5      1     3       2    7    9
         2     9       2    4    8
         3    10       0    2    7
  6      1     3       0    8    6
         2     3       9    8    4
         3     6       0    6    4
  7      1     6       6    7    8
         2     7       0    6    8
         3     8       0    5    8
  8      1     4       5    6    8
         2     4       2    7    8
         3     7       0    5    5
  9      1     1       0    5    5
         2     2       9    4    4
         3     4       6    2    4
 10      1     3       1    9    6
         2     8       0    4    5
         3     8       0    5    4
 11      1     2       5    4    7
         2     3       0    4    6
         3     8       0    1    6
 12      1     3       7    3    5
         2     3       0    4    5
         3     8       0    2    5
 13      1     4       5    8    4
         2     4       6    7    5
         3     7       4    6    1
 14      1     5       0    7    8
         2     7       4    5    8
         3     9       0    5    8
 15      1     1       9    3    8
         2     5       9    2    6
         3     7       8    2    6
 16      1     1       6    3    3
         2     1       5    4    3
         3     5       3    3    3
 17      1     8       4    4   10
         2     9       0    3    5
         3     9       0    4    4
 18      1     0       0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  N 1  N 2
   26   75   94
************************************************************************
