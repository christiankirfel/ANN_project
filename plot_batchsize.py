import matplotlib.pyplot as plt

#batchSizesCPUTF2 = []
#timeSecondsCPUTF2 = []

#files = ['benchResultsCPU_8c.txt','benchResultsGPU.txt']
files = ['benchResultsCPU_8c.txt','benchResultsCPU_16c.txt']

batchSizes = [[] for _ in range(len(files))]
timeSeconds = [[] for _ in range(len(files))]

for i,fi in enumerate(files):
	with open(fi,'r') as f:
		for line in f:
			tempList = line.split(' ')
			batchSizes[i].append(int(tempList[0]))
			timeSeconds[i].append(float(tempList[1].replace('\n','')))

print(len(batchSizes))

#with open('batchsize_CPU_TF2.log','r') as f:
#    for line in f:
#        tempList = line.split(' ')
#        batchSizesCPUTF2.append(int(tempList[0]))
#        timeSecondsCPUTF2.append(float(tempList[1].replace('\n','')))

perHour = [[] for _ in range(len(files))]

for i,fi in enumerate(files):
	perHour[i] = [360000/j for j in timeSeconds[i]]
#perHourCPUTF2 = [36000/i for i in timeSecondsCPUTF2]

for i in range(len(perHour)):
   print(str(batchSizes[i]) + ' ' + str(perHour[i]) + '\n')

plt.style.use('fast')
#plt.plot(batchSizesGPU, timeSecondsGPU,'ro')
#plt.plot(batchSizesCPU, timeSecondsCPU,'bo')
for i,fi in enumerate(files):
	plt.plot(batchSizes[i], perHour[i],f'C{i}+')
	plt.plot(batchSizes[i], perHour[i],f'C{i}--',linewidth=0.5)
#plt.plot(batchSizesCPUTF2,perHourCPUTF2,'go')
#plt.annotate('',xytext=(8192,50),xy=(4300,10),arrowprops=dict(facecolor='black',width=0.1,headwidth=4,headlength=5))
#plt.annotate('6.72',xy=(9000,50))
#plt.title('Performance with different batch sizes')
plt.xlabel('Batch size')
plt.xscale('log')
batchSizes=[128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144]
plt.xticks(batchSizes, [str(2**x) for x in range(7,19)], rotation = 'vertical')
plt.ylim(bottom=0.)
#plt.ylabel('Time (s) for 10 iterations')
plt.ylabel('Epochs per hour')
plt.grid(True,'major','y')
#plt.legend(('CPU (8 cores)',None,'GPU',''))
plt.legend(('CPU (8 cores)',None,'CPU (16 cores)',None))
plt.subplots_adjust(bottom=0.2)
plt.savefig('benchResults.png')