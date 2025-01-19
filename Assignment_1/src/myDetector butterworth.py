import wfdb, os, subprocess
from Helper_functions import exponential_weight_preprocessing, detection_stage
from scipy.signal import butter, filtfilt

# Read the record (replace 's20011' with your actual file name, without extension)
LTST_folder_path = r'C:\Users\katar\Documents\FRI\OBSS\Seminarske naloge\long-term-st-database-1.0.0\long-term-st-database-1.0.0'
MIT_folder_path = r'C:\Users\katar\Documents\FRI\OBSS\Seminarske naloge\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0'
output_folder_path = r'C:\Users\katar\Documents\FRI\OBSS\Seminarske naloge\Output_LTST_butter' 

records = []
folder_path = LTST_folder_path #change

# Read the RECORDS file
with open(os.path.join(folder_path, 'RECORDS'), 'r') as f:
    for line in f:
        line = line.strip()
        records.append(line)
#print(len(records))

# For butterworth filter
def butterworth_processing(ecg_signal, fs):
    order = 4
    cutoff = [0.1, 50]  
    nyquist = fs / 2  # Nyquist frequency
    normalized_cutoff = [c / nyquist for c in cutoff]
    b, a = butter(order, normalized_cutoff, btype='band', analog=False)
    filtered_signal = filtfilt(b, a, ecg_signal)
    return filtered_signal

# Konstante
fs = 250 # za LTST
#fs = 360  # za MIT
RRmin = int(0.2 * fs)  
QRSint = int(0.06 * fs)  
N = RRmin + QRSint
alpha = 1 - 2 / (N -1)

for record_name in records:
    dat_record = wfdb.rdrecord(os.path.join(folder_path, record_name))

    # Extract the signal into a numpy array
    signal = dat_record.p_signal
    ecg_signal = signal[:, 0] # LOAD the ECG signal here

    # Butterworth filter
    butter_signal = butterworth_processing(ecg_signal, fs)

    ecg_signal = butter_signal

    # run preprocessing and detection
    EW_mean, EW_var = exponential_weight_preprocessing(ecg_signal, alpha)
    _, _ = detection_stage(EW_var[:10], fs, QRSint, RRmin)
    r_peak_positions, thresholds_list = detection_stage(EW_var, fs, QRSint, RRmin)

    # make an asc file
    with open(os.path.join(output_folder_path, f'{record_name}.asc'), 'w') as f:
        for peak in r_peak_positions:
            f.write(f"0:00:00.00 {peak} N 0 0 0\n")

    print(f'Done with {record_name}')

    # transform asc into qrs file
    convert_asc_qrs = f"wrann -r {record_name} -a qrs < {record_name}.asc"
    subprocess.run(convert_asc_qrs, cwd=output_folder_path, shell=True)

    # compare qrs to atr with bxb
    bxb_command = f'bxb -r {record_name} -a atr qrs -l eval1.txt eval2.txt -f 0'
    subprocess.run(bxb_command, cwd=output_folder_path, shell=True)

# sumstats on eval files
sumstats_command = f'sumstats eval1.txt eval2.txt >results.txt'
subprocess.run(sumstats_command, cwd=output_folder_path, shell=True)
print('Done with sumstats')

