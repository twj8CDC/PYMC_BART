
import pandas as pd
import numpy as np
import scipy.stats as sp
import sksurv as sks
import sksurv.metrics as skm
import sksurv.linear_model

# import surv_bart as bmb

# import surv_bart as bmb
from surv_bart_pkg import surv_bart as bmb



import matplotlib.pyplot as plt

# survival utilities
def get_x_matrix(N=100,
                x_vars=1,
                VAR_CLASS = None,
                VAR_PROB = None,
                rng = None
):
    if rng is None:
        rng = np.random.default_rng(seed=99)
        print(rng.random(1))
    
    bern = sp.bernoulli
    bern.random_state = rng

    # create an x matrix
    x_mat = np.zeros((N, x_vars))
    for idx, x in enumerate(VAR_CLASS):
        if x == 2:
            # x1 = sp.bernoulli.rvs(VAR_PROB[idx], size = N)
            # x1 = bern.rvs(VAR_PROB[idx], size = N)
            x1 = rng.binomial(1, VAR_PROB[idx], size = N)
        else:
            x1 = rng.uniform(0, VAR_CLASS[idx], size = N)
            x1 = np.round(x1, 3)
        x_mat[:,idx] = x1
    if x_vars > len(VAR_CLASS):
        for idx in np.arange(len(VAR_CLASS), x_vars):
            x_mat[:,idx] = rng.uniform(0,1,size=N)
    return x_mat

def sim_surv(
    x_mat = None,
    lambda_f=None, 
    alpha_f = None, 
    eos = None,
    cens_scale = None,
    time_scale = None,
    return_full = False,
    true_only = False,
    rng = None
):
    N = x_mat.shape[0]
    if rng is None:
        rng = np.random.default_rng(seed=99)

    
    # lambda and alpha
    lmbda = eval(lambda_f)
    if "x_mat" not in alpha_f:
        a = np.repeat(eval(alpha_f), N)
    else:
        a = eval(alpha_f)
    if not true_only:
        # unif = np.random.uniform(size=N)
        unif = rng.uniform(size=N)
        tmp = np.power((-1*np.log(unif)).reshape(-1,1), (1/a).reshape(-1,1))
        tlat = (tmp/lmbda.reshape(-1,1)).reshape(-1,)
        # censor
        if cens_scale is not None:
            # cens = np.ceil(np.random.exponential(size = N, scale = cens_scale))
            cens = np.ceil(rng.exponential(size = N, scale = cens_scale))
            t_event  = np.minimum(cens, np.ceil(tlat))
            status = (tlat <= cens) * 1
        else:
            # cens=np.zeros(N)
            t_event = np.ceil(tlat)
            status = np.ones(N)
        
        # eos censoring
        if eos is not None:
            eos_msk = t_event > eos
            t_event[eos_msk] = eos
            status[eos_msk] = 0

        # get event times
        t_max = int(t_event.max())
        t = np.linspace(0,t_max+1, t_max+2)
    else:
        t = np.linspace(0,eos+1, eos+2)
    # survival and hazard
    sv_mat = np.exp(-1 * np.power((lmbda.reshape(-1,1)*t), a.reshape(-1,1)))
    hz_mat = (lmbda * a).reshape(-1,1) * np.power(lmbda.reshape(-1,1)*t, a.reshape(-1,1)-1)

    # scale
    if time_scale is not None:
        s0 = sv_mat.shape[1]
        t_scale = np.arange(0,s0+1, time_scale)
        sv_mat_scale = sv_mat[:,t_scale]
        hz_mat_scale = hz_mat[:,t_scale]
        t_scale2 = t_scale/time_scale

    if return_full:
        if true_only:
            return {"sv_true":sv_mat, "hz_true":hz_mat, "t":t}, {"sv_true":sv_mat_scale, "hz_true":hz_mat_scale, "t":t_scale, "t2":t_scale2}
        return t_event, status, x_mat, {"sv_true":sv_mat, "hz_true":hz_mat, "t":t}, {"sv_true":sv_mat_scale, "hz_true":hz_mat_scale, "t":t_scale, "t2":t_scale2}
    if true_only:
        return {"sv_true":sv_mat_scale, "hz_true":hz_mat_scale, "t":t_scale, "t2":t_scale2}
    return t_event, status, x_mat,{"sv_true":sv_mat_scale, "hz_true":hz_mat_scale, "t_scale":t_scale}        
    

def get_cph(y_sk, x_sk, pred):
    cph = sks.linear_model.CoxPHSurvivalAnalysis().fit(x_sk, y_sk)
    coef = {"exp coef": np.exp(cph.coef_)}
    cph_sv = cph.predict_survival_function(pred)
    cph_sv = np.array([fx(fx.x) for fx in cph_sv])
    cph_chz = cph.predict_cumulative_hazard_function(pred)
    cph_chz = np.array([fn(fn.x) for fn in cph_chz])
    return coef, cph_sv, cph_chz

def get_sv_metrics(sv, prob, y_sk, y_sk_train=None):
    if y_sk_train is None:
        y_sk_train = y_sk
    bart_cindex = []
    for i in np.arange(prob.shape[1]):
        ci = skm.concordance_index_censored(y_sk["Status"], y_sk["Survival_in_days"], prob[:,i])
        bart_cindex.append(np.round(ci[0],3))

    
    l = np.unique(y_sk["Survival_in_days"])
    ibs = skm.integrated_brier_score(y_sk_train, y_sk, sv[:,0:-1], l[:-1])
    bs = skm.brier_score(y_sk_train, y_sk, sv[:,0:-1], l[:-1])
    bs = [f.tolist() for f in bs]
    return {"cindex": bart_cindex, "bs": bs, "ibs":np.round(ibs, 4)}

def quick_kpm_plot(y_sk, msk, cph_sv, sv):
    fig, ax = plt.subplots(1)
    cov_mask = msk
    kpm_all = sks.nonparametric.kaplan_meier_estimator(y_sk["Status"], y_sk["Survival_in_days"])
    kpm_cov = sks.nonparametric.kaplan_meier_estimator(y_sk["Status"][cov_mask], y_sk["Survival_in_days"][cov_mask])
    kpm_ncov = sks.nonparametric.kaplan_meier_estimator(y_sk["Status"][~cov_mask], y_sk["Survival_in_days"][~cov_mask])


    ax.step(kpm_cov[0], kpm_cov[1], label="kpm_cov", color="green")
    ax.step(kpm_ncov[0], kpm_ncov[1], label="kpm_ncov", color="lightgreen")
    ax.step(kpm_all[0], cph_sv[cov_mask].mean(0), color = "blue", label="cph_cov")
    ax.step(kpm_all[0], cph_sv[~cov_mask].mean(0), color = "lightblue", label="cph_ncov")
    ax.step(kpm_all[0], sv[cov_mask].mean(0), color = "red", label = "b_cov")
    ax.step(kpm_all[0], sv[~cov_mask].mean(0), color = "pink", label= "b_ncov")

    ax.legend()
    return fig

def get_true_rmse_bias(true, pred, time_col):
    true = true[:,time_col]
    pred = pred[:, time_col]
    rmse = np.round(np.sqrt(np.mean(np.power(true - pred, 2), axis=0)), 4)
    bias = np.round(np.mean(true - pred, axis = 0), 4)
    return {"rmse":rmse.tolist(), "bias":bias.tolist(), "time_col":time_col}


def quick_kpm_true(x_mat, status, t_event, true, true_scale):
    fig, ax = plt.subplots(1)
    # plots for comparing the true vs kpm estimates and scaled 
    cov_mask = (x_mat[:,0]==1)
    y_sk = bmb.get_y_sklearn(status, t_event)

    kpm_all = sks.nonparametric.kaplan_meier_estimator(y_sk["Status"], y_sk["Survival_in_days"])
    kpm_cov = sks.nonparametric.kaplan_meier_estimator(y_sk["Status"][cov_mask], y_sk["Survival_in_days"][cov_mask])
    kpm_ncov = sks.nonparametric.kaplan_meier_estimator(y_sk["Status"][~cov_mask], y_sk["Survival_in_days"][~cov_mask])

    # plot estimate from draws
    ax.step(kpm_all[0], kpm_all[1], label="all_kpm")
    ax.step(kpm_cov[0], kpm_cov[1], label="covid_kpm")
    ax.step(kpm_ncov[0], kpm_ncov[1], label="ncovid_kpm")
    # plot true
    ax.plot(true["t"], true["sv_true"].mean(0), label = "all_true")
    ax.plot(true["t"], true["sv_true"][cov_mask,:].mean(0), label = "cov_true")
    ax.plot(true["t"], true["sv_true"][~cov_mask,:].mean(0), label = "ncov_true")

    ax.step(true_scale["t"], true_scale["sv_true"].mean(0), label = "all_scl")
    ax.step(true_scale["t"], true_scale["sv_true"][cov_mask,:].mean(0), label = "cov_scl")
    ax.step(true_scale["t"], true_scale["sv_true"][~cov_mask,:].mean(0), label = "ncov_scl")

    # plt.xlim(0,400)
    ax.legend()
    return fig


def quick_kpm_true_scale(x_mat, status, t_event, true_scale, time_scale):
    fig, ax = plt.subplots(1)
    # plots comparing kpm estimate of scaled time compared to the true scaled time
    t_event2 = bmb.get_time_transform(t_event, time_scale=time_scale)
    y_sk = bmb.get_y_sklearn(status, t_event2)

    cov_mask = (x_mat[:,0]==1)
    kpm_all = sks.nonparametric.kaplan_meier_estimator(y_sk["Status"], y_sk["Survival_in_days"])
    kpm_cov = sks.nonparametric.kaplan_meier_estimator(y_sk["Status"][cov_mask], y_sk["Survival_in_days"][cov_mask])
    kpm_ncov = sks.nonparametric.kaplan_meier_estimator(y_sk["Status"][~cov_mask], y_sk["Survival_in_days"][~cov_mask])

    ax.step(kpm_all[0], kpm_all[1], label="all", alpha=0.2)
    ax.step(kpm_cov[0], kpm_cov[1], label="covid", alpha=0.2)
    ax.step(kpm_ncov[0], kpm_ncov[1], label="ncovid", alpha=0.2)


    ax.step(true_scale["t2"], true_scale["sv_true"].mean(0), alpha=0.2, label="all_scl")
    ax.step(true_scale["t2"], true_scale["sv_true"][cov_mask,:].mean(0), alpha=0.2, label = "cov_scl")
    ax.step(true_scale["t2"], true_scale["sv_true"][~cov_mask,:].mean(0), alpha=0.2, label = "ncov_scl")
    ax.set_title("KPM Estimate of scaled times w/ True scaled")
    ax.legend()

    plt.step(true_scale["t2"], true_scale["sv_true"].mean(0), alpha=0.2, label="all_scl")
    plt.step(true_scale["t2"], true_scale["sv_true"][cov_mask,:].mean(0), alpha=0.2, label = "cov_scl")
    plt.step(true_scale["t2"], true_scale["sv_true"][~cov_mask,:].mean(0), alpha=0.2, label = "ncov_scl")
    plt.title("KPM Estimate of scaled times w/ True scaled")
    plt.legend()

def sv_plot(sv, y_sk_coh, msk, y_sk, cc1, strat=True, cred_int=True, kpm_all=True, kpm_sample=True, whr = "mid"):
    t = np.arange(1,sv["mt_m"].shape[0]+1)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.step(t, sv["mt_m"], label="covid", where=whr)
    ax.step(t, sv["mf_m"], label="non-covid", where=whr)
    if cred_int:
        ax.fill_between(t, sv["mt_q"][0], sv["mt_q"][1], step=whr, alpha=0.1)
        ax.fill_between(t, sv["mf_q"][0], sv["mf_q"][1], step=whr, alpha=0.1)
    if kpm_sample:
        km_cov = sks.nonparametric.kaplan_meier_estimator(y_sk_coh["Status"][msk], y_sk_coh["Survival_in_days"][msk])
        km_ncov = sks.nonparametric.kaplan_meier_estimator(y_sk_coh["Status"][~msk], y_sk_coh["Survival_in_days"][~msk])
        ax.step(km_cov[0], km_cov[1], label="km_cov_smp", where=whr)
        ax.step(km_ncov[0], km_ncov[1], label="km_ncov_smp", where=whr)
    if kpm_all:
        msk2 = cc1[:,7] == 1
        km_cov_all = sks.nonparametric.kaplan_meier_estimator(y_sk["Status"][msk2], y_sk["Survival_in_days"][msk2], conf_type="log-log")
        km_ncov_all = sks.nonparametric.kaplan_meier_estimator(y_sk["Status"][~msk2], y_sk["Survival_in_days"][~msk2], conf_type="log-log")
        ax.step(km_cov_all[0], km_cov_all[1], label="km_cov_all", where=whr)
        # ax.fill_between(km_cov_all[0], km_cov_all[2][0], km_cov_all[2][1], step="pre", alpha=0.1)
        ax.step(km_ncov_all[0], km_ncov_all[1], label="km_ncov_all", where=whr)
        # ax.fill_between(km_ncov_all[0], km_ncov_all[2][0], km_ncov_all[2][1], step="pre", alpha=0.1)
    ax.legend()
    ax.set_xticks([1,2,3,4])
    return fig

def calib_metric_internal(sv, y_sk_coh, t, q = np.arange(0,1,.1)):
    sv = sv.mean(0)
    qt = np.quantile(sv[:,t-1], q, axis=0)
    l = len(qt)
    obs_out = np.zeros((qt.shape[0], sv.shape[1]))
    pred_out = np.zeros((qt.shape[0],sv.shape[1]))
    diff_out = np.zeros((qt.shape[0],sv.shape[1]))
    for idx,lo in enumerate(qt):
        if idx <= l-2:
            up = qt[idx+1]
        else:
            up = 1.1
        tmp = np.where(((sv[:,t-1] >= lo) & (sv[:,t-1] < up)))
        obs2 = y_sk_coh[tmp]
        kp = sks.nonparametric.kaplan_meier_estimator(obs2["Status"], obs2["Survival_in_days"])
        obs = np.round(kp[1],3)
        pred = np.round(sv[tmp].mean(0),3)
        # fix uneven obs pred len
        if len(obs) != len(pred):
            sdf = np.setdiff1d([1,2,3,4],kp[0])
            print(f"adjusting at time {sdf}")
            obs = np.insert(obs, sdf, pred[sdf])
        diff = np.round(obs-pred,3)
        obs_out[idx,:] = obs
        pred_out[idx,:] = pred
        diff_out[idx, :] = diff
    return obs_out, pred_out, diff_out, qt

def calib_metric(sv, y_sk_coh, q=np.arange(0,1,0.1)):
    times = sv.shape[2]
    obs_out = np.zeros((times, q.shape[0], times))
    pred_out = np.zeros((times, q.shape[0], times))
    diff_out = np.zeros((times, q.shape[0], times))
    qt_out = []

    for i in np.arange(times):
        obs, pred, diff, qt = calib_metric_internal(sv = sv, y_sk_coh = y_sk_coh, t=i+1, q = q)
        obs_out[i,:,:] = obs
        pred_out[i,:,:] = pred
        diff_out[i,:,:] = diff
        # print(qt)
        qt_out.append(qt)
    return {"obs":obs_out, "pred":pred_out, "diff":diff_out, "qt":qt_out}

def plot_calib_diff(trn_calib):
    fig, axs = plt.subplots(2,2, figsize=(10,10))
    axs[0,0].plot(trn_calib["diff"][0].T, "o", label=np.round(trn_calib["qt"][0], 2))
    axs[0,0].legend()
    axs[0,1].plot(trn_calib["diff"][1].T, "o", label=np.round(trn_calib["qt"][1], 2))
    axs[0,1].legend()
    axs[1,0].plot(trn_calib["diff"][2].T, "o", label=np.round(trn_calib["qt"][2], 2))
    axs[1,0].legend()
    axs[1,1].plot(trn_calib["diff"][3].T, "o", label=np.round(trn_calib["qt"][3], 2))
    axs[1,1].legend()
    return fig