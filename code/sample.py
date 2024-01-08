import os
postfix = 0
for file in os.listdir('predictions/'):
    num = int(file.split('_')[1].split('.')[0])
    if num>postfix:
        postfix = num
postfix += 1
filename = 'predictions/output_{}.txt'.format(postfix)

print(filename)