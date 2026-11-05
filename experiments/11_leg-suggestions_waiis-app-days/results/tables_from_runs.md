## GPS candidates (SHL User 1, 20 s; six-class F1 after fusion + HMM)

```
                    0      1      2   mean  e1_six_f1  six_f1_before_hmm  gps_alone_six_f1_covered  covered_official  rawlegs_edits_per_h
gps_rule                                                                                                                                 
C1              0.953  0.955  0.956  0.955      0.964              0.857                     0.703             0.755                0.903
C2              0.955  0.957  0.956  0.956      0.966              0.837                     0.680             0.582                0.912
C3              0.953  0.952  0.954  0.953      0.962              0.855                     0.693             0.755                0.923
C4              0.954  0.954  0.955  0.954      0.964              0.856                     0.703             0.755                0.902
C5              0.949  0.949  0.952  0.950      0.960              0.843                     0.601             0.764                1.006
no_gps          0.949  0.951  0.951  0.950      0.966              0.788                       NaN             0.000                0.972
ref_minute_1hz  0.953  0.958  0.955  0.955      0.963              0.855                     0.710             0.739                0.928
```

## Leg grid, seed 0 (raw HMM legs vs the objective's choice)

```
                   edits_per_h  leg_f1  leg_f1_top1  time_acc  wrong_unflagged_time  flagged_leg_share  range_top1  range_top2  range_ece
model data legs                                                                                                                          
E2    shl  raw           0.909   0.823        0.823     0.952                 0.048              0.000       0.912       0.957      0.049
           chosen        0.896   0.824        0.824     0.952                 0.048              0.000       0.912       0.957      0.049
      es   raw           1.398   0.718        0.718     0.917                 0.083              0.000       0.808       0.943      0.118
           chosen        1.384   0.718        0.722     0.916                 0.083              0.010       0.808       0.943      0.118
E1    shl  raw           0.843   0.830        0.830     0.965                 0.035              0.000       0.921       0.960      0.042
           chosen        0.838   0.836        0.838     0.964                 0.035              0.003       0.921       0.960      0.042
      es   raw           1.404   0.718        0.718     0.915                 0.085              0.000       0.808       0.942      0.119
           chosen        1.461   0.705        0.709     0.915                 0.085              0.010       0.808       0.942      0.119
```

## 20 s boundary refinement (SHL User 1, 1 s truth)

```
       minute_boundary_median_min  refined_boundary_median_min  minute_edits_per_h  refined_edits_per_h  moved
model                                                                                                         
E2                          0.346                        0.555               0.940                0.955  568.0
E1                          0.327                        0.487               0.866                0.881  578.0
```

## WAIIS days (strict matching, IoU >= 0.5)

```
                  minutes  legs_model  legs_gps_only_still  legs_gps_only_moving  legs_unknown_low_confidence  diary_matched  diary_legs  extra_legs  run_seconds_total
date       model                                                                                                                                                       
2026-10-30 E1       138.0        13.0                  4.0                   3.0                          0.0            5.0        10.0         2.0              0.181
           E2       138.0        11.0                  4.0                   3.0                          0.0            2.0        10.0         2.0              0.178
2026-10-31 E1       119.0        10.0                  4.0                   5.0                          0.0            0.0        11.0         1.0              0.130
           E2       119.0        10.0                  4.0                   5.0                          0.0            0.0        11.0         1.0              0.124
2026-11-03 E1       160.0        21.0                  5.0                   4.0                          0.0            3.0         7.0         1.0              0.161
           E2       160.0        21.0                  5.0                   4.0                          0.0            3.0         7.0         2.0              0.155
```

## WAIIS days, lenient comparison (core minutes)

```
model                               E2      E1  package
vehicle_legs                    19.000  19.000   19.000
vehicle_right                    6.000   8.000    5.000
vehicle_found                    6.000   8.000    5.000
vehicle_wrong                    8.000   6.000    9.000
vehicle_no_data                  5.000   5.000    5.000
vehicle_core_minutes_with_data  83.000  83.000   83.000
vehicle_core_minute_accuracy     0.699   0.807    0.542
other_legs                       9.000   9.000    9.000
other_right                      4.000   4.000    5.000
other_found                      4.000   4.000    5.000
other_wrong                      1.000   1.000    0.000
other_no_data                    4.000   4.000    4.000
other_core_minutes_with_data    23.000  23.000   23.000
other_core_minute_accuracy       0.739   0.739    0.826
extra_vehicle_legs               0.000   0.000    0.000
```

### per mode

```
model              E2    E1  package
bike_legs         3.0   3.0      3.0
bike_right        3.0   3.0      3.0
bike_no_data      0.0   0.0      0.0
car_legs          1.0   1.0      1.0
car_right         1.0   1.0      1.0
car_no_data       0.0   0.0      0.0
bus_legs          1.0   1.0      1.0
bus_right         1.0   1.0      1.0
bus_no_data       0.0   0.0      0.0
rail_legs        14.0  14.0     14.0
rail_right        1.0   3.0      0.0
rail_no_data      5.0   5.0      5.0
walking_legs      8.0   8.0      8.0
walking_right     4.0   4.0      5.0
walking_no_data   3.0   3.0      3.0
running_legs      1.0   1.0      1.0
running_right     0.0   0.0      0.0
running_no_data   1.0   1.0      1.0
```

## SHL Users 2-3 (once)

```
model                                 E2     E1
c1_bag_edits_per_h                 0.875  0.975
c1_bag_leg_f1                      0.863  0.858
c1_bag_leg_recall_rail             0.867  0.933
c1_bag_wrong_unflagged_time        0.069  0.069
c1_hand_edits_per_h                1.475  1.500
c1_hand_leg_f1                     0.764  0.751
c1_hand_leg_recall_rail            0.667  0.733
c1_hand_wrong_unflagged_time       0.190  0.149
c1_hips_edits_per_h                1.075  1.000
c1_hips_leg_f1                     0.830  0.853
c1_hips_leg_recall_rail            0.800  0.933
c1_hips_wrong_unflagged_time       0.074  0.065
c1_torso_edits_per_h               1.175  1.175
c1_torso_leg_f1                    0.793  0.816
c1_torso_leg_recall_rail           0.800  0.867
c1_torso_wrong_unflagged_time      0.129  0.110
minute_bag_edits_per_h             0.850  0.725
minute_bag_leg_f1                  0.858  0.886
minute_bag_leg_recall_rail         0.800  1.000
minute_bag_wrong_unflagged_time    0.072  0.072
minute_hand_edits_per_h            1.400  1.450
minute_hand_leg_f1                 0.785  0.767
minute_hand_leg_recall_rail        0.800  0.800
minute_hand_wrong_unflagged_time   0.192  0.148
minute_hips_edits_per_h            1.025  0.950
minute_hips_leg_f1                 0.823  0.861
minute_hips_leg_recall_rail        0.800  0.933
minute_hips_wrong_unflagged_time   0.077  0.059
minute_torso_edits_per_h           0.825  1.100
minute_torso_leg_f1                0.863  0.812
minute_torso_leg_recall_rail       0.867  0.867
minute_torso_wrong_unflagged_time  0.107  0.102
```

