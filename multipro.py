from multiprocessing import Process,Lock
import time
import reverbration_test_mp_01
import numpy as np
class Run(Process):
    def __init__(self,name,sleeptime,lock):
        super().__init__()
        self.name=name
        self.sleeptime = sleeptime
        self.lock = lock
    def run(self):




        time.sleep(self.sleeptime)
        spike,spike_array,voltage = reverbration_test_mp_01.run()
        name_array = 'Ach_array_'+self.name + '.txt'
        name_spike = 'Ach_spike_' + self.name+ '.txt'
        np.savetxt(name_array, spike_array)
        np.savetxt(name_spike, spike)
        self.lock.acquire()
        print('%s runing end' %self.name)

        self.lock.release()

p1=Run('1',1)
p2=Run('2',2)
p3=Run('3',3)
p4=Run('4',4)
p5 = Run('5',5)
p1.start() #start会自动调用run
p2.start()
p3.start()
p4.start()
p5.start()

p1.join()
p2.join()
p3.join()
p4.join()
p5.join()

