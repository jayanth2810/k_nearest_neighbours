__author__ = 'jayanthvenkataraman'

import knn
import matplotlib
#matplotlib.use('qt4agg')

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
dating_matrix,dating_labels = knn.file_to_matrix("datingTestSet.txt")
ax.scatter(dating_matrix[:,1], dating_matrix[:,2])
#ax.axis([-2,25,-0.2,2.0])
#plt.xlabel('Percentage of Time Spent Playing Video Games')
#plt.ylabel('Liters of Ice Cream Consumed Per Week')
plt.show()