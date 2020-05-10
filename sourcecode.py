#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:58:54 2019

@author: ranahamzaintisar
"""

import numpy as np
import matplotlib.pyplot as plt

'''defining a k-means alogithm function where we pass the dataset as 2-d image vector array with the required number of clusters(n)
returns final n codebook vectors as the mean of the vector points lying in the codebook vector's cluster'''

def kmeans_algorithm (dataset, n):

    # create a copy of the  2-d image vecotr array
    data_copy = dataset.copy()

    # shuffe 2-d image vector arrays along the first axis(row)
    np.random.shuffle(data_copy)

    # take the first n image vector arrays, from the shuffeld 2-d image vector array, as initial random codebook vector assignments
    codebook = data_copy[:n]

    # Compute the eucledian disance between vector arrays in dataset and the randomly selected codebook vectors

    # substract each codebook vector from the dataset image vectors.
    # numpy broadcasting allows to substract all dataset image vectors form the codebook vector array even if their shape dont match.
    # Step 1: extend the codebook vector array by adding a new dimension in between the two existing dimensions
    # extending a new dimension allows us to use the rule of broadcasting- array of unequal dimension are when compatable if one the array dimension is 1 here the extended dimension is 1.
    extend_codebook = codebook[:,np.newaxis]
    # Step 2: substract extended codebook vector array form image vector array
    difference = dataset - extend_codebook


    #find the absolute distance from the difference, abs distance = ||difference||
    abs_dist_extended = np.sqrt((difference)**2)

    #reduce the 3-d absolute distance array back into a 2-d array
    abs_dist = abs_dist_extended.sum(axis=2)

    # compute an array of index for each vector in the dataset; the index value will be the nearest index of the nearest codebook vector from the data image vector.
    nearest_codebook = np.argmin(abs_dist,axis=0)


    #assign new codebook vectors, as mean of the dataset image vectors that lie closest to a particular codebook vector assigned above
    new_codebook = np.array([dataset[nearest_codebook==i].mean(axis=0) for i in range(codebook.shape[0])])

    return new_codebook




'''importing the dataset and indexing the dataset to select data points for digit 7'''

dataset = np.loadtxt("mfeat-pix.txt")
data_seven = dataset[1400:1599]



'''part one of sloution with k=1'''

#run the clustering alorithm

kmenas_output = kmeans_algorithm(data_seven,1)


# plotting images in the  coded in the codebook vector
plt.figure()
plt.imshow(kmenas_output.reshape(16, 15),cmap="binary")



'''part two of sloution with k=2'''

#run the clustering alorithm

kmenas_output_2 = kmeans_algorithm(data_seven,2)
print(kmenas_output_2[0].shape)


# plotting images in the  coded in the codebook vector
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(kmenas_output_2[0].reshape(16, 15),cmap="binary")
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(kmenas_output_2[1].reshape(16, 15),cmap="binary")

'''part three of sloution with k=3'''

#run the clustering alorithm

kmenas_output_3 = kmeans_algorithm(data_seven,3)

# plotting images in the  coded in the codebook vector
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(kmenas_output_3[0].reshape(16, 15),cmap="binary")
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(kmenas_output_3[1].reshape(16, 15),cmap="binary")
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(kmenas_output_3[2].reshape(16, 15),cmap="binary")


'''part four of sloution with k=200'''

#run the clustering alorithm

kmenas_output_200 = kmeans_algorithm(data_seven,200)

# plotting images in the  coded in the codebook vector
c=1
fig = plt.figure(figsize=(15,15))
for i in range(0,200,20):
   ax1 = fig.add_subplot(5,2,c)
   ax1.imshow(kmenas_output_200[i].reshape(16, 15),cmap="binary")
   c=c+1



'''exploring clustering with all of the in the dataset digits'''



full_dataset = np.loadtxt("mfeat-pix.txt")

'''with k=10'''



kmenas_output_full = kmeans_algorithm(full_dataset,10)


# plotting images in the  coded in the codebook vector
c=1
fig = plt.figure(figsize=(10,10))
for i in range(0,10):
   ax1 = fig.add_subplot(5,2,c)
   ax1.imshow(kmenas_output_full[i].reshape(16, 15),cmap="binary")
   c=c+1


'''with k=20'''



kmenas_output_full = kmeans_algorithm(full_dataset,20)


# plotting images in the  coded in the codebook vector
c=1
fig = plt.figure(figsize=(10,10))
for i in range(0,20):
   ax1 = fig.add_subplot(10,2,c)
   ax1.imshow(kmenas_output_full[i].reshape(16, 15),cmap="binary")
   c=c+1


'''with k=30'''



kmenas_output_full = kmeans_algorithm(full_dataset,30)


# plotting images in the  coded in the codebook vector
c=1
fig = plt.figure(figsize=(10,10))
for i in range(0,30):
   ax1 = fig.add_subplot(15,2,c)
   ax1.imshow(kmenas_output_full[i].reshape(16, 15),cmap="binary")
   c=c+1
