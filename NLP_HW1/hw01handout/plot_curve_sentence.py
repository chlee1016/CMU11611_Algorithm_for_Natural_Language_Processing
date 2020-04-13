import matplotlib.pyplot as plt
import numpy as np

plt.figure()
x = (np.arange(10)+1) * 10
# print(x)
y_word = np.array([0.1114, 0.0834, 0.0746, 0.0665, 0.0613, 0.0600, 0.0605, 0.0568, 0.0559, 0.0541])
y_sentence = np.array([0.8176, 0.7629, 0.7429, 0.7211, 0.7023, 0.6888, 0.6776, 0.6688, 0.6594, 0.6576])
# plt.plot(x,y_word)
# plt.title('title',pad=10)
plt.xlabel('The number of training data (%)',labelpad=10)
plt.ylabel('The error rate by sentence',labelpad=10)

plt.plot(x,y_sentence, linewidth=2, color='r')
plt.xticks(np.arange(10,110,10))
plt.grid()
plt.show()