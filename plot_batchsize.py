import matplotlib.pyplot as plt

batchSizes = []
timeSeconds = []

with open('batchsize.log','r') as f:
    for line in f:
        tempList = line.split(' ')
        batchSizes.append(int(tempList[0]))
        timeSeconds.append(float(tempList[1].replace('\n','')))

#print(batchSizes)
#print(timeSeconds)

plt.plot(batchSizes, timeSeconds,'ro')
plt.title('Performance with different batch sizes')
plt.xlabel('Batch size')
plt.xscale('log')
plt.xticks(batchSizes, [str(2**x) for x in range(7,19)], rotation = 'vertical')
plt.ylabel('Time (s) for 10 iterations')
plt.grid(True,'major','y')
plt.subplots_adjust(bottom=0.2)
plt.savefig('performance_batchsize.png')