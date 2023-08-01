# PH: 𝛼 = 2.0, 𝜆 = exp{3 + 0.1(x1 + x2 + x3 + x4 + x5 + x6) + x7}
#S(t|𝛼, 𝜆) = e ^−(t∕𝜆)𝛼
def sim_surv(N=100, 
            T=100, 
            x_vars = 1, 
            lambda_f=None, 
            a=2, 
            alpha_f = None, 
            seed=999, 
            cens_ind = True,
            cens_scale = 20,
            err_ind = False):
    # np.random.seed(seed)

    x_mat = np.zeros((N, x_vars))
    for x in np.arange(x_vars):
        x1 = sp.bernoulli.rvs(.5, size = N)
        x_mat[:,x] = x1
    # calculate lambda
    
    # set lambda
    if lambda_f is None:
        lmbda = np.exp(2 + 0.3*(x_mat[:,0] + x_mat[:,1]) + x_mat[:,2])
    else:
        lmbda = eval(lambda_f)
    
    # set alpha if specified
    if alpha_f is None:
        a = np.repeat(a, N)
    else:
        a = eval(alpha_f)

    # add error
    if err_ind:
        error = sp.norm.rvs(0, .5, size = N)
        lmbda=lmbda + error

    # get time series
    t = np.linspace(0,T, T)

    # calculate survival and event times
    sv_mat = np.zeros((N, t.shape[0]))
    tlat = np.zeros(N)
    for idx, l in enumerate(lmbda):
        sv = np.exp(-1 * np.power((t/l), a[idx]))
        sv_mat[idx,:] = sv
        
        # generate event times 
        unif = np.random.uniform(size=1)
        ev = lmbda[idx] * np.power((-1 * np.log(unif)), 1/a[idx])
        tlat[idx] = ev

    if cens_ind:
        # censor
        cens = np.ceil(np.random.exponential(size = N, scale = cens_scale))

        # min cen and surv event
        t_event  = np.minimum(cens, np.ceil(tlat))
        status = (tlat <= cens) * 1
    else:
        cens=np.zeros(N)
        t_event = np.ceil(tlat)
        status = np.ones(N)

        

    return sv_mat, x_mat, lmbda, a, tlat, cens, t_event, status



def get_x_info(x_mat):
    x = np.unique(x_mat, axis=0, return_index=True, return_counts=True)
    x_out, x_idx, x_cnt = x[0], x[1], x[2]
    return x_out, x_idx, x_cnt

def get_status_perc(status):
    out = status.sum()/status.shape[0]
    cens = 1-out
    return out, cens

def get_event_time_metric(t_event):
    t_mean = t_event.mean()
    t_max = t_event.max()
    return t_mean, t_max

def get_train_matrix(x_mat, t_event, status):
    et = pd.DataFrame({"status": status, "time":t_event})
    train = pd.concat([et, pd.DataFrame(x_mat)],axis=1)
    return train

def get_y_sklearn(status, t_event):
    y = np.array(list(zip(np.array(status, dtype="bool"), t_event)), dtype=[("Status","?"),("Survival_in_days", "<f8")])
    return y

def plot_sv(x_mat, sv_mat, t, title="TITLE", save=False, dir=".", show=False):
    dist_x, dist_idx = np.unique(x_mat, axis=0, return_index=True)
    if type(t) == int:
        print("here")
        tt = np.arange(t)
    else:
        tt = t

    # print(tt)
    try:
        fig = plt.figure()
        if len(sv_mat) != len(dist_idx):
            for idx, i in enumerate(sv_mat[dist_idx]):
                plt.plot(tt, i, label = str(dist_x[idx]))
                plt.legend()
                plt.title(title)
        else:
            for idx, i in enumerate(sv_mat):
                # plt.step(i.x, i.y, label = str(dist_x[idx]))
                plt.plot(tt, i, label = str(dist_x[idx]))
                plt.legend()
                plt.title(title)
        if show:
            plt.show()
        if save:
            plt.savefig(f"{dir}/{title}.png")
    finally:
        plt.close(fig)

def surv_pre_train2(data_x_n, data_y, X_TIME=True):
    # set up times
    # t_sort = np.append([0], np.unique(data_y["Survival_in_days"]))
    t_sort = np.unique(data_y["Survival_in_days"])
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
    
    
    t_out, delta_out, pat_x_out = np.concatenate(t_out), np.concatenate(delta_out), np.concatenate(pat_x_out)
    if X_TIME:
        pat_x_out = np.array([np.concatenate([np.array([t_out[idx]]), i]) for idx, i in enumerate(pat_x_out)])
    return t_out, delta_out, pat_x_out

def get_bart_test(x_out, T):
    s0 = x_out.shape[0]
    s1 = x_out.shape[1]
    # create time range
    # d1 = np.arange(T+1)
    d1 = T
    # repeating time range
    d2 = np.tile(d1,s0).reshape(d1.shape[0]*s0,1)
    # repeat x_out and shape as long by nvar
    d3 = np.tile(x_out, d1.shape[0]).reshape(s0*d1.shape[0], s1)
    # holding matrix
    d4 = np.matrix(np.zeros((d3.shape[0], d3.shape[1] + 1))) # always +1 because only adding on time col
    # replace
    d4[:,0] = d2
    d4[:,1:(s1+1)] = d3
    return d4