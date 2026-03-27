MODEL_CONFIG = {
    'Walmsley2020': {
        "rf_features": [
            'enmoTrunc', 'enmoAbs', 'mean', 'sd', 'coefvariation', 'median',
            'min', 'max', '25thp', '75thp', 'autocorr', 'fmax', 'pmax',
            'fmaxband', 'pmaxband', 'entropy', 'fft1', 'fft2', 'fft3', 'fft4',
            'fft5', 'fft6', 'fft7', 'fft8', 'fft9', 'fft10', 'MAD', 'MPD',
            'skew', 'kurt', 'f1', 'p1', 'f2', 'p2', 'f625', 'p625',
            'totalPower'
        ],
        "rf_params": {
            "n_estimators": 200,
            "min_samples_leaf": 5e-5,
            "max_depth": 20,
            "random_state": 42,
            "replacement": True,
            "sampling_strategy": "not minority",
            "n_jobs": 4
        },
        "labels": {
            "sleep": "Sleep",
            "sedentary": "Sedentary Behaviour",
            "light": "Light Activity",
            "moderate-vigorous": "Moderate-Vigorous Activity",
        },
        "actinet_params": {
            "hmm_handle_sleep_transitions": True,
            "hmm_ignore_transition_gaps": False,
        }
    },
    'Willetts2018': {
        "rf_features": [
            'enmoTrunc', 'enmoAbs', 'xMean', 'yMean', 'zMean', 'xRange',
            'yRange', 'zRange', 'xStd', 'yStd', 'zStd', 'xyCov', 'xzCov',
            'yzCov', 'mean', 'sd', 'coefvariation', 'median', 'min', 'max',
            '25thp', '75thp', 'autocorr', 'corrxy', 'corrxz', 'corryz',
            'avgroll', 'avgpitch', 'avgyaw', 'sdroll', 'sdpitch', 'sdyaw',
            'rollg', 'pitchg', 'yawg', 'fmax', 'pmax', 'fmaxband', 'pmaxband',
            'entropy', 'fft1', 'fft2', 'fft3', 'fft4', 'fft5', 'fft6', 'fft7',
            'fft8', 'fft9', 'fft10', 'MAD', 'MPD', 'skew', 'kurt', 'f1', 'p1',
            'f2', 'p2', 'f625', 'p625', 'totalPower'
        ],
        "rf_params": {
            "n_estimators": 500,
            "min_samples_leaf": 1e-5,
            "max_depth": 20,
            "random_state": 42,
            "replacement": True,
            "sampling_strategy": "not minority",
            "n_jobs": 4
        },
        "labels": {
            "sleep": "Sleep",
            "sit-stand": "Sit/Stand",
            "vehicle": "Vehicle",
            "walking": "Walking",
            "mixed": "Mixed Activity",
            "bicycling": "Bicycling",
        },     
        "actinet_params": {
            "hmm_handle_sleep_transitions": False,
            "hmm_ignore_transition_gaps": False,
        }
    }
}
