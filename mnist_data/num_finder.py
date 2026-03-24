from mnist_utils import Mnist

#load mnist dataset
path = 'mnist_data' #download mnist dataset to this folder
_, Y_test_raw = Mnist.load_mnist(path, kind='t10k')

def num_finder(num, samples):
  idxs = []
  for i in range(0,len(Y_test_raw)):
    if Y_test_raw[i]  == num:
      idxs.append(i)
      af = idxs[0:samples]
  print(af)
num_finder(5, 10)

# 10 of each number
#0: [3, 10, 13, 25, 28, 55, 69, 71, 101, 126]
#1: [2, 5, 14, 29, 31, 37, 39, 40, 46, 57]
#2: [1, 35, 38, 43, 47, 72, 77, 82, 106, 119]
#3: [18, 30, 32, 44, 51, 63, 68, 76, 87, 90]
#4: [4, 6, 19, 24, 27, 33, 42, 48, 49, 56]
#5: [8, 15, 23, 45, 52, 53, 59, 102, 120, 127]
#6: [11, 21, 22, 50, 54, 66, 81, 88, 91, 98]
#7: [0, 17, 26, 34, 36, 41, 60, 64, 70, 75]
#8: [61, 84, 110, 128, 134, 146, 177, 179, 181, 184]
#9: [7, 9, 12, 16, 20, 58, 62, 73, 78, 92]

# tricky images
#0: 
#1: 2221
#2: 
#3: 4443
#4: 
#5: 8, 9970
#6: 
#7: 
#8: 
#9: 833