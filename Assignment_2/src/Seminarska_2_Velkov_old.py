import wfdb, os, subprocess, numpy as np
from numba import njit
from sampen import sampen2
from scipy import stats

current_path = os.getcwd()

RECORDS_path = r'C:\Users\katar\Documents\FRI\OBSS\Seminarska naloga 2\term-preterm-ehg-database-1.0.1\RECORDS'
files_path = r'C:\Users\katar\Documents\FRI\OBSS\Seminarska naloga 2\term-preterm-ehg-database-1.0.1\tpehgdb'

RECORDS = []
with open(RECORDS_path, 'r') as f:
    for line in f:
        line = line.strip().split('/')
        RECORDS.append(line[1])

files = os.listdir(files_path)

# calculate sample entropy
@njit
def get_sample_entropy(signal, r, m):
    N = len(signal)
    r = r * np.std(signal)

    A=0 # of matches of length m+1
    B=0 # of matches of length m

    for i in range(N-m-1):
        for j in range(i+1, N-m-1):
            match = 1
            for k in range(0, m):
                if np.abs(signal[i+k]-signal[j+k]) > r:
                    match = 0
                    break
            if match == 1:
                B += 1
                if np.abs(signal[i+m+1]-signal[j+m+1]) < r:
                    A += 1
    if A == 0 or B == 0:
        sample_entropy = - np.log((N-m) / (N-m-1))
    else:
        sample_entropy = - np.log(A/B)
    return sample_entropy


four_groups = {'PE': [], 'PL': [], 'TE': [], 'TL': []}
results = {'PE': [], 'PL': [], 'TE': [], 'TL': []}
counter = 0
for file_name in RECORDS:
    # read the header file:
    hea_record = wfdb.rdheader(os.path.join(files_path, file_name))
    print(hea_record.record_name)
    # da ločiš na 4 skupine rabis : pregnancy duration (gestation);
    #gestation duration at the time of recording;
    comments = hea_record.comments
    gestation = float(comments[2].split(' ')[-1])
    gestation_duration = float(comments[3].split(' ')[-1])

    # read the dat file:
    dat_samp = wfdb.rdsamp(os.path.join(files_path, file_name))
    dat_record = wfdb.rdrecord(os.path.join(files_path, file_name))
    dat_signal = dat_record.p_signal

    # calculate sample entropy for a certain channel:
    channel=9
    #signal = dat_signal[:, channel]
    # cut first and last 3600 samples:
    signal = dat_signal[3600:-3600, channel]
    print('Signal shape:', signal.shape)
    _ = get_sample_entropy(signal[:10], 0.15, 3) # for numba
    
    sample_entropy = get_sample_entropy(signal, 0.15, 3)

    # classify to four groups:
    # PE (gestation < 37 weeks (preterm), gestation duration < 26 weeks)
    if gestation < 37 and gestation_duration < 26:
        four_groups['PE'].append(file_name)
        results['PE'].append(sample_entropy)
    # PL (gestation < 37 weeks (preterm), gestation duration >= 26 weeks)
    elif gestation < 37 and gestation_duration >= 26:
        group='PL'
        four_groups['PL'].append(file_name)
        results['PL'].append(sample_entropy)

    # TE (gestation >= 37 weeks (term), gestation duration < 26 weeks)
    elif gestation >= 37 and gestation_duration < 26:
        group='TE'
        four_groups['TE'].append(file_name)
        results['TE'].append(sample_entropy)

    # TL (gestation >= 37 weeks (term), gestation duration >= 26 weeks)
    else: #gestation >= 37 and gestation_duration >= 26:
        group='TL'
        four_groups['TL'].append(file_name)
        results['TL'].append(sample_entropy)
    

    print('Signal:', counter, 'group:', group, 'Sample entropy:', sample_entropy)
    counter += 1


# save results
for key, values in results.items():
    np.save(f'{key}_results_channel_{channel}_cut_signal.npy', values)   

    with open(f'{key}_results_channel_{channel}_cut_signal.txt', 'w') as f:
        for value in values:
            f.write(f'{value}\n')