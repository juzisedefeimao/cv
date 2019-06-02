import matplotlib.pyplot as plt
import numpy as np

with open('D:\\jjj\\zlrm\\logs\\zlrm_relation_net_accuracy.txt', 'r') as f:
    data_list = f.readlines()

x = []
y = []

for i in range(len(data_list)):
    if data_list[i] != ' ':
        data_line = data_list[i].split(' ')
        x.append(int(data_line[0]))
        y.append(float(data_line[2].split('\\')[0]))

x = np.array(x)
y = np.array(y)
# print(x)
# print(y)

# plt.figure(1)
# plt.subplot(211)
plt.plot(x,y)

plt.savefig('classifier_accuracy_1.png')
plt.show()

