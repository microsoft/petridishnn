import numpy as np
import time
import random
import multiprocessing as mp


def fake_train_nn(instance):
    total_time = random.uniform(0, 10)
    time.sleep(total_time)
    print(f'Process {instance} completed in {total_time} seconds.')    
    


if __name__ == "__main__":

    for i in range(10):
        mp.Process(target=fake_train_nn, args=(i,)).start()
    
    print('Non blocking. Yay!')
    
    





