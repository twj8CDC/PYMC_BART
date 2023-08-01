import numpy as np

def surv_pre_bart(times, delta, x_train=[]):
    N = len(times)
    if N != len(delta):
        return
    
    events = np.sort(np.unique(times))
    K = len(events)


    
    time_long = list()
    k = 0
    for i in times:
        tmp_event = np.zeros(K)
        for j in events:
            if events[j] <= times[i]:
                tmp_event[j] = delta[j]*(time[i] == events[j])
        time_long[k] = tmp_event
        k = k+1
