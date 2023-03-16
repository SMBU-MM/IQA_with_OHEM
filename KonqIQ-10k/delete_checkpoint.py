import os
paths = ['balanced_0.7_woa_resnet34']#os.listdir('./')

for path in paths:
    for i in range(1,11):
        for j in range(0,7):
            model_name = '{}-{:0>5d}.pt'.format('DataParallel', j)
            p = "./{}/{}/checkpoint/{}".format(path, i, model_name)
            print(p)
            if os.path.exists(p):
                os.remove(p)
                print('True:', p)
        for j in range(8,12):
            model_name = '{}-{:0>5d}.pt'.format('DataParallel', j)
            p = "./{}/{}/checkpoint/{}".format(path, i, model_name)
            print(p)
            if os.path.exists(p):
                os.remove(p)
                print('True:', p)