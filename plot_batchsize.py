import matplotlib.pyplot as plt

batchSizesGPU = []
timeSecondsGPU = []

batchSizesCPU = []
timeSecondsCPU = []

batchSizesCPUTF2 = []
timeSecondsCPUTF2 = []

with open('batchsize.log','r') as f:
    for line in f:
        tempList = line.split(' ')
        batchSizesGPU.append(int(tempList[0]))
        timeSecondsGPU.append(float(tempList[1].replace('\n','')))

with open('batchsize_CPU.log','r') as f:
    for line in f:
        tempList = line.split(' ')
        batchSizesCPU.append(int(tempList[0]))
        timeSecondsCPU.append(float(tempList[1].replace('\n','')))

with open('batchsize_CPU_TF2.log','r') as f:
    for line in f:
        tempList = line.split(' ')
        batchSizesCPUTF2.append(int(tempList[0]))
        timeSecondsCPUTF2.append(float(tempList[1].replace('\n','')))

perHourGPU = [36000/i for i in timeSecondsGPU]
perHourCPU = [36000/i for i in timeSecondsCPU]
perHourCPUTF2 = [36000/i for i in timeSecondsCPUTF2]

#plt.plot(batchSizesGPU, timeSecondsGPU,'ro')
#plt.plot(batchSizesCPU, timeSecondsCPU,'bo')
plt.plot(batchSizesGPU, perHourGPU,'ro')
plt.plot(batchSizesCPU, perHourCPU,'bo')
plt.plot(batchSizesCPUTF2,perHourCPUTF2,'go')

plt.title('Performance with different batch sizes')
plt.xlabel('Batch size')
plt.xscale('log')
plt.xticks(batchSizesGPU, [str(2**x) for x in range(7,19)], rotation = 'vertical')
#plt.ylabel('Time (s) for 10 iterations')
plt.ylabel('Iterations per hour')
plt.grid(True,'major','y')
plt.legend(('GPU','CPU (TF1)','CPU (TF2)'))
plt.subplots_adjust(bottom=0.2)
plt.savefig('performance_batchsize.png')