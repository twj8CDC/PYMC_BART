def get_auc(event, time, hz, w):
    t_out = np.zeros(shape=time.shape)
    for ti in range(0,len(time)):
        #set time
        t = time[ti]
        # print("t", t)

        # get numerator
        auc_num = 0
        pat_rng = range(0, len(hz))
        # sum over patients: sum_rank_ipcw
        # where sum_rank_ipcw: rank_indicator * ipcw weight
        # where rank_indicator: 1 if the other patient has event after time, current patient has event before/at time and other patients hazard is < current patients
            # eg. if time is T = 10, current patient event is 5 and hz is 3, and other patients events are 15,20,1; hz 0,1,4
            # then for current patient iterations we would see
                # iter | ot_pat > t | cur_pat <= t | hz(ot_pat) <= hz(cur_pat) | indicator | weight | out | cumsum
                # 1    | 1          | 1            | 1                         | 1         | 1.3    | 1.3 |  1.3
                # 2    | 1          | 1            | 1                         | 1         | 1.3    | 1.3 |  2.6
                # 3    | 0          | 1            | 0                         | 0         | 1.3    | 0   |  2.6 ***
        # The cumsum for each patient will be added together to form the numerator
        for i in pat_rng:
            auc_num1 = 0
            for j in pat_rng:
                # 1 if iter pat time > t
                i1 = 1 if event[j] > t else 0
                # 1 if hold pat time <= t
                i2 = 1 if event[i] <= t else 0
                # 1 if hz iter pat <= hz hold pat
                i3 = 1 if hz[j] <= hz[i] else 0
                # if iter pat is > t, hold pat <= t and iter pat hz <= hz hold pat then all are 1 and you adjust by the ipcw 
                auc_num1 += i1 * i2 * w[i] * i3 
            # print("auc_num1", auc_num1)
            auc_num += auc_num1

        # denom
        # the denom is the cumsum I(pat_time > t) * cumsum I(pat_time <= t * wi)
        auc_denom = 0
        den_i1 = 0
        for i in pat_rng:
            i1 = 1 if event[i] > t else 0
            den_i1 += i1

        den_i2 = 0
        for i in pat_rng:
            i1 = 1 if event[i] <= t else 0
            # added in w to the equation
            den_i2 += i1 * w[i]

        # eval auc
        # the auc final calc for each timepoint is then ratio of number correctly classified over number expected
        auc = auc_num / (den_i1 * den_i2)
        t_out[ti] = auc

    return t_out


def get_brier(y_time, delta, times, surv, ipcw, kipcw, ig = False, ar_out = False):
    t_out = np.zeros(times.shape)
    n = len(y_time)

    for ti, t in enumerate(times):
        print(t)
        # t = times[ti]
        score = 0
        for i in range(0, len(y_time)):
            sv = surv[i](t)
            
            i1 = 1 if y_time[i] <= t and delta[i] == 1 else 0 
            surv_ipcw1 = pow((0 - sv), 2)/ipcw[i] 

            i2 = 1 if y_time[i] > t else 0
            surv_ipcw2 = pow((1 - sv), 2)/kipcw[ti]

            score += (i1 * surv_ipcw1) + (i2 * surv_ipcw2)
        t_out[ti] = score/n
    
    if ig == False:
        return t_out
    else:
        ig_brier = np.trapz(t_out, times)/(times[-1]-times[0])
        if ar_out:
            return ig_brier, t_out
        else:
            return ig_brier
