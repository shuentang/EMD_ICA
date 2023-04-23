from PyEMD import CEEMDAN, EMD
import numpy as np
import pylab as plt
import pickle as pickle
from mne.preprocessing import ICA
from sklearn.decomposition import FastICA

import EntropyHub as EH
import numpy as np
def SampleEntropy2(Datalist, r=0.2, m=2):
	th = r * np.std(Datalist) #容限阈值
	return EH.SampEn(Datalist,m,r=th)[0][-1]


def read_data(filename):
    x = pickle._Unpickler(open(filename, 'rb'))
    x.encoding = 'latin1'
    data = x.load()
    return data

# file = 's12.dat'
# sub = read_data(file)
# d = sub['data']
# eeg = d[0][0]
# print('eeg shape is ',eeg.shape)

f = open("fp2.txt", "r")
res = f.readlines()
t = list()
for cur in res:
    # strip方法去除每一行的换行符
    t.append(float(cur.strip(' \n')))
# t为一个二维列表
f.close()

eeg = np.array(t)
eeg = eeg - min(eeg)
eeg = eeg - min(eeg)
s = eeg
N = len(s)
fs = 512
t = np.arange(0,N)/fs



ceemdan = EMD()
IMF = ceemdan.emd(s)



print("CEEMDAN  shape is ", IMF.shape )
# (9, 3584)


P = IMF.shape[0]+1


# Plot results
plt.figure(1)
plt.subplot(P,1,1)
plt.plot(t, s, 'r')
plt.title("Input signal")
plt.xlabel("Time [s]")

for n, imf in enumerate(IMF):
    plt.subplot(P,1,n+2)  
    plt.plot(t, imf, 'g')
    plt.title("IMF "+str(n+1))
    plt.xlabel("Time [s]")


plt.figure(2)
plt.subplot(P,1,1)
plt.plot(t, s, 'r')
plt.title("Input signal")
plt.xlabel("Time [s]")

ica = FastICA()  
imf_data =  IMF.T           
res = ica.fit_transform(imf_data)

# TODA   API 中X(n_Sample, n_feature)
# 这里的shape 是 （3584，9），  不可能是（9， 3584）

print("Fast Ica  shape is ", res.shape )
En_ls = []  # 样本熵的 list形式
di_En = {}  # 样本熵的字典形式
di_ica = {} # ica的 字典形式

rdata = res.T

for n, imf in enumerate(rdata):
    plt.subplot(P,1,n+2)
    plt.plot(t, imf, 'g')
    di_ica[n] = imf
    plt.title("Fast ICA "+str(n+1))
    di_En[n] = SampleEntropy2(imf)
    En_ls.append(SampleEntropy2(imf))
    print("SamEn is ", SampleEntropy2(imf))
    plt.xlabel("Time [s]")


# 对字典进行排序
sort_t_En= sorted(di_En.items(), key = lambda kv:(kv[1], kv[0]))
# 排序完是tuple 元组模式， 后改成字典模式
sort_di_En = {}
sort_ls_En = []
for key, value in sort_t_En:
    sort_di_En[key] = value
    sort_ls_En.append(value)


sum2 = []
zero = np.zeros(len(imf)).tolist()
for  key, value in di_ica.items():
     if( di_En[key] < 0.4):
          sum2.append(zero)
     else:
          sum2.append(value)
sum2 = np.array(sum2).T
print('sum2 shape is ',sum2.shape)

invser = ica.inverse_transform(sum2) 
# API 中X(n_Sample, n_feature)
print('inverse transform shape is ', invser.shape)
inverse_ica = np.array(invser)


# Plot results
plt.figure(3)
plt.subplot(P,1,1)
plt.plot(t, s, 'r')
plt.title("Input signal")
plt.xlabel("Time [s]")

i_data = inverse_ica.T

for n, imf in enumerate(i_data):
    plt.subplot(P,1,n+2)  
    plt.plot(t, imf, 'g')
    plt.title("ICA to IMF "+str(n+1))
    plt.xlabel("Time [s]")


# Plot results
plt.figure(4)
plt.subplot(2,1,1)

data = np.sum(inverse_ica, axis=1)
print('data shape is ',data.shape)
plt.plot(t, data, 'r')
plt.title("Answer signal")
plt.xlabel("Time [s]")

plt.subplot(2,1,2)
plt.plot(t, s, 'r')
plt.title("Input signal")
plt.xlabel("Time [s]")

print("MSE is ", 0.5*np.sum((data-s)**2/len(data)) )
plt.savefig('CEEMDAN_example')

plt.show()