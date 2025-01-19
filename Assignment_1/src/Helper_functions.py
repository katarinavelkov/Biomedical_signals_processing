import numpy as np
from numba import njit

def exponential_weight_preprocessing(signal, alpha):
    EW_mean = np.zeros_like(signal)
    EW_var = np.zeros_like(signal)
    
    for i in range(1, len(signal)):
        EW_mean[i] = (1 - alpha) * signal[i] + alpha * EW_mean[i - 1]
        EW_var[i] = (1 - alpha) * (EW_var[i - 1] + alpha * (signal[i] - EW_mean[i - 1]) ** 2) # tega rabis za detekcijo uporabit
    
    return EW_mean, EW_var

@njit
def detection_stage(EW_signal, fs, QRSint, RRmin):
    decay_factor = -np.log(0.1) / (0.4 * fs)
    # instatiate the threshold:
    th = 0.05
    # prepare the variables for the FSM
    r_peak_positions = []
    state = 3
    counter = 0
    thresholds_list = []

    for n in range(len(EW_signal)):

        if state == 1:
            counter += 1
            if EW_signal[n] > th:
                th = EW_signal[n]
            if counter > RRmin + QRSint:
                window = RRmin + QRSint
                EW_signal_window = EW_signal[n - window : n] 
                max_peak_position = n - window + np.argmax(EW_signal_window)
                r_peak_positions.append(max_peak_position)
                counter = n - max_peak_position # counter = d
                state = 2
 
        elif state == 2:
            counter += 1
            if counter > RRmin:
                state = 3
                th = np.mean(np.array([EW_signal[peak] for peak in r_peak_positions]))

        elif state == 3:
            th = th * np.exp(-decay_factor)

            if EW_signal[n] > th:
                state = 1
                counter = 0
                th = EW_signal[n]

        thresholds_list.append(th)
    return r_peak_positions, thresholds_list