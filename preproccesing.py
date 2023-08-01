def surv_pre_train(data_x_n, data_y):
    # set up times
    t_sort = np.append([0], np.unique(data_y["Survival_in_days"]))
    t_ind = np.arange(0,t_sort.shape[0])
    t_dict = dict(zip(t_sort, t_ind))

    # set up delta
    delta = np.array(data_y["Status"], dtype = "int")
    

    t_out = []
    pat_x_out = []
    delta_out = []
    for idx, t in enumerate(data_y["Survival_in_days"]):
        # get the pat_time and use to get the array of times for the patient
        p_t_ind = t_dict[t]
        p_t_set = t_sort[0:p_t_ind+1]
        t_out.append(p_t_set)
        
        size = p_t_set.shape[0]
        # get patient array
        pat_x = np.tile(data_x_n.iloc[idx].to_numpy(), (size, 1))
        pat_x_out.append(pat_x)

        # get delta
        pat_delta = delta[idx]
        delta_set = np.zeros(shape=size, dtype=int)
        delta_set[-1] = pat_delta
        delta_out.append(delta_set)
    
    return np.concatenate(t_out), np.concatenate(delta_out), np.concatenate(pat_x_out)


def surv_pre_test(data_x_n, data_y):
    t_sort = np.append([0], np.unique(data_y["Survival_in_days"]))
    t_out = []
    pat_x_out = []
    for idx, t in enumerate(data_y["Survival_in_days"]):
        # get the pat_time and use to get the array of times for the patient
        p_t_set = t_sort
        t_out.append(p_t_set)
        
        size = p_t_set.shape[0]
        # get patient array
        pat_x = np.tile(data_x_n.iloc[idx].to_numpy(), (size, 1))
        pat_x_out.append(pat_x)
    
    return np.concatenate(t_out),  np.concatenate(pat_x_out)
    