import time
from threading import Timer

"""
Heartbeat/progress crap

Philly requires printing PROGRESS on stdout, otherwise jobs are killed after 14400 sec.
The progress must be increasing.

To work around this on petridish, we have the following function that repeatly cheat 
the progress bar with small inc. 

"""

def print_fake_progress(*args):
    wa = args[0]
    curr = time.time()
    if curr - wa.heartbeat >= wa.interval - 1:
        # subtract 1 above to be safe
        wa.new_heart_beat(is_fake=True)
        wa.print_progress_percent()
        print("PhillyHeartBeatWorkAround.n_fake={}".format(wa.n_fake))
    if wa.not_over():
        Timer(wa.interval, print_fake_progress, args=[wa]).start()


class PhillyHeartBeatWorkAround(object):
    
    def __init__(self, interval=14300, cnt_inc=0.2, max_cnt=10000):
        self.interval = interval
        self.cnt = 0
        self.max_cnt = max_cnt
        self.n_fake = 0
        self.cnt_inc = cnt_inc
        self.heartbeat = time.time()
        if self.not_over():
            Timer(self.interval, print_fake_progress, args=[self]).start()

    def new_heart_beat(self, cnt=None, is_fake=False):
        if is_fake:
            self.n_fake += 1
        else:
            self.cnt = cnt
        self.heartbeat = time.time()
        
    def print_progress_percent(self):
        pp = (float(self.cnt) + self.cnt_inc * self.n_fake) / self.max_cnt * 100.
        print("\nPROGRESS: {0:07.7f}%\n".format(pp))

    def not_over(self):
        return self.cnt + self.n_fake * self.cnt_inc < self.max_cnt