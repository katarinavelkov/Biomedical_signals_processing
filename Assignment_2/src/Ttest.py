import numpy as np
from scipy import stats

PE = np.load('PE_results_channel_9.npy')
PL = np.load('PL_results_channel_9.npy')
TE = np.load('TE_results_channel_9.npy')
TL = np.load('TL_results_channel_9.npy')

PE_cut = np.load('PE_results_channel_9_cut_signal.npy')
PL_cut = np.load('PL_results_channel_9_cut_signal.npy')
TE_cut = np.load('TE_results_channel_9_cut_signal.npy')
TL_cut = np.load('TL_results_channel_9_cut_signal.npy')

P = np.concatenate((PE, PL))
T = np.concatenate((TE, TL))

P_cut = np.concatenate((PE_cut, PL_cut))
T_cut = np.concatenate((TE_cut, TL_cut))


# test variance of the groups
print('PE var:', np.var(PE), 'PL var:', np.var(PL), 'TE var', np.var(TE), 'TL var', np.var(TL))
print('group P:', np.var(P), 'group T:', np.var(T))

# ttest
# term and preterm EARLY
t1, p1 = stats.ttest_ind(TE, PE, equal_var=False)
print('T-test results p1:', p1)

# term and preterm LATE
t2, p2 = stats.ttest_ind(TL, PL, equal_var=False)
print('T-test results p2:', p2)

# term and preterm both groups
t6, p6 = stats.ttest_ind(P, T, equal_var=False)
print('T-test results p6:', p6)

# term and preterm EARLY cut signal
t1_cut, p1_cut = stats.ttest_ind(TE_cut, PE_cut, equal_var=False)
print('T-test results p1_cut:', p1_cut)

# term and preterm LATE cut signal
t2_cut, p2_cut = stats.ttest_ind(TL_cut, PL_cut, equal_var=False)    
print('T-test results p2_cut:', p2_cut)

# term and preterm both groups cut signal
t6_cut, p6_cut = stats.ttest_ind(P_cut, T_cut, equal_var=False)
print('T-test results p6_cut:', p6_cut)