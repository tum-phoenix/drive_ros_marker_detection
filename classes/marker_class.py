# -*- coding: utf-8 -*-
# mapping: id (as int) to sign description (string)
import pickle

# all carolo cup signs currently in use
marker_name_dict = {
        0: '10_speed_limit',
        1: '20_speed_limit',
        2: '30_speed_limit',
        3: '40_speed_limit',
        4: '50_speed_limit',
        5: '60_speed_limit',
        6: '70_speed_limit',
        7: '80_speed_limit',
        8: '90_speed_limit',
        9: 'end_10_speed_limit',
        10: 'end_20_speed_limit',
        11: 'end_30_speed_limit',
        12: 'end_40_speed_limit',
        13: 'end_50_speed_limit',
        14: 'end_60_speed_limit',
        15: 'end_70_speed_limit',
        16: 'end_80_speed_limit',
        17: 'end_90_speed_limit',
        18: 'right_arrow',
        19: 'left_arrow',
        20: 'startline',
        21: 'broken_crossing_line',
        22: 'continuous_crossing_line',
        23: 'left_crossing_turning_line',
        24: 'right_crossing_turning_line',
        25: 'startline',
        26: 'zebra_crossing'
}

with open('marker_name_dict.pkl', 'wb') as f:
    pickle.dump(marker_name_dict, f)
