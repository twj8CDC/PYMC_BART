# Databricks notebook source
# MAGIC %md
# MAGIC # Objective
# MAGIC - Demonstrate the effect of downsampling and matched cohorts have on outcomes

# COMMAND ----------

# %pip install scikit-survival
# %pip install pymc_experimental
# %pip install matplotlib colorcet
# %pip install --force-reinstall pymc
%pip install pymc_bart

# COMMAND ----------

from pathlib import Path
# import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import importlib
# import mlflow as ml

import utillities as ut
import surv_bart as bmb

# COMMAND ----------

# MAGIC %md
# MAGIC There are two connected py files for this workflow. 
# MAGIC - Utilities provides the simulation generator fx and additional evaluation metrics
# MAGIC - surv_bart (bmb) is the bart model wrapper. With an sklearn like api it conveniently extends the bart model for sv settings

# COMMAND ----------

importlib.reload(ut)
importlib.reload(bmb)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a simulation dataset

# COMMAND ----------

# generage the random number generator object
rng = np.random.default_rng(seed=990)

# COMMAND ----------

# MAGIC %md
# MAGIC Simulation is based on a weibull model with a lambda and alpha parameters. It generates true survival/hazard fxs as well as random event/status times from the inverse of the weibull pdf. 
# MAGIC
# MAGIC The covariate matrix is generated from as number of observations (N), number of variables (x_vars) and class of variables beings binary or continuous. Probability of draws for the var_class ==2 (binary) is given by the corresponding index of the var_prob. 
# MAGIC
# MAGIC Lambda and alpha are calculated from the linear equation defined around the x_mat covariate matrix. 
# MAGIC
# MAGIC eos and time_scale represent an end of study marker and a event time scaling to reduce time points.
# MAGIC
# MAGIC Returns:
# MAGIC - t_event is event time
# MAGIC - status is 1 if event occurs or 0 if censored
# MAGIC - x_mat is the covariate matrix
# MAGIC - true is the true sv/hz at times
# MAGIC - true_scale is the true sv/hz at times scaled to the t_event scale

# COMMAND ----------

N = 10000
X_VARS = 10
VAR_CLASS = [2,10,3,1,2]
VAR_PROB = [0.5, 0.5, 0.5, 0.5, 0.5]
LAMBDA = "np.exp(-5 + .2*x_mat[:,0] + 0.01*np.log(x_mat[:,1]+0.00001) + 0.2*(x_mat[:,2] + x_mat[:,3] + x_mat[:,4]))" 
ALPHA_F = "3 + .1*x_mat[:,0]"
eos = 120
time_scale=20


# get SV
x_mat = ut.get_x_matrix(N=N, x_vars=X_VARS, VAR_CLASS=VAR_CLASS, VAR_PROB=VAR_PROB, rng=rng)
t_event, status, x_mat, true, true_scale = ut.sim_surv(x_mat, LAMBDA, ALPHA_F, eos, time_scale=time_scale, return_full=True, rng = rng)

# COMMAND ----------

# MAGIC %md
# MAGIC A test dataset can be created using the simulation fxs with the same parameters as the train dataset. The rng object instantiated above allows for reproducible sampling with a set seed.

# COMMAND ----------

# test
t_x_mat = ut.get_x_matrix(N=N, x_vars=X_VARS, VAR_CLASS=VAR_CLASS, VAR_PROB=VAR_PROB, rng=rng)
t_t_event, t_status, t_x_mat, t_true, t_true_scale = ut.sim_surv(t_x_mat, LAMBDA, ALPHA_F, eos, time_scale=time_scale, return_full=True, rng=rng)

# COMMAND ----------

# MAGIC %md
# MAGIC To check the the simulated datasets are appropriate comparison of the simulated KPM output scaled and uncaled to the true sv times is completed below. 
# MAGIC
# MAGIC With a high number of observations it is apparent that the simulated datasets KPM stratified on the first covariate (which has been named "covid") for correspondence with the actual proposed analyses appropriately approximates the true sv curves.

# COMMAND ----------

ut.quick_kpm_true(x_mat, status, t_event, true, true_scale)

# COMMAND ----------

# MAGIC %md
# MAGIC Similarily the test set evaluation is completed below.

# COMMAND ----------

# test comparison
ut.quick_kpm_true(t_x_mat, t_status, t_t_event, t_true, t_true_scale)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally the comparison of the scaled times and the true scaled times are completed below. Demonstrating that scaling the event times does not disrupt the estimator or properties of the sv curve.

# COMMAND ----------

ut.quick_kpm_true_scale(x_mat, status, t_event, true_scale, time_scale)

# COMMAND ----------

# test
ut.quick_kpm_true_scale(t_x_mat, t_status, t_t_event, t_true_scale, time_scale)

# COMMAND ----------

# MAGIC %md
# MAGIC The proposed work is focused on evaluating a main variable ("covid") and this corresponding simulation follows this example w/ the first covariate being a binary variable representing covid status. 
# MAGIC
# MAGIC Simple frequencies of the distributions of covid status and corresponding event status are shown below.

# COMMAND ----------

mc_mask = (x_mat[:,0] == 1)
train_dict = {
    "N":status.shape,
    "minor class cases" :status[mc_mask].shape[0],
    "minor class events": status[mc_mask].sum(),
    "minor class ev prop": status[mc_mask].sum()/status[mc_mask].shape[0],
    "major class cases":status[~mc_mask].shape[0],
    "major class events" : status[~mc_mask].sum(),
    "major class ev prop": status[~mc_mask].sum()/status[~mc_mask].shape[0]
}

mc_mask = (t_x_mat[:,0] == 1)
test_dict = {
    "N":status.shape,
    "minor class cases" :t_status[mc_mask].shape[0],
    "minor class events": t_status[mc_mask].sum(),
    "minor class ev prop": t_status[mc_mask].sum()/t_status[mc_mask].shape[0],
    "major class cases":t_status[~mc_mask].shape[0],
    "major class events" : t_status[~mc_mask].sum(),
    "major class ev prop": t_status[~mc_mask].sum()/t_status[~mc_mask].shape[0]
}

# log these
print(train_dict)
print(test_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC # Surv Bart Model

# COMMAND ----------

# MAGIC %md
# MAGIC The Surv Bart model is self contained in the surv_bart.py script and access to the prior/model/posteriro fx are accessible as bmb.fx.
# MAGIC
# MAGIC The first step is to complete simple transformation of the event_time/status datasets to achieve a long-form sv dataset that can be passed to the bart model. 
# MAGIC
# MAGIC Steps:
# MAGIC 1. get_time_transform - scales the event times to coarsen the data
# MAGIC     - this allows for more efficient computation as the long-form data entered into the model is of a length n*k were k is the number of distinct time points.
# MAGIC 2. get_y_sklearn - joins the status and event_times together into a array of tuples that follow the schema for the scikit_survival package.
# MAGIC 3. get_case_cohort - used only in the case of class imbalanced cases (<10%). This returned dataset is a case-cohort sample.
# MAGIC     - when prop = 1, it returns an identity matrix with the y_sk, x_mat and weights of 1.
# MAGIC 4. surv_pre_train - creates the long-form training matrix. coh_x now contains the extended time in the first column.
# MAGIC 5. get_posterior_test - generates a longform test dataset for predictions through the full length of the survival dataset. 
# MAGIC     - Note that in this first case it is generated on the training dataset.
# MAGIC

# COMMAND ----------

t_event2 = bmb.get_time_transform(t_event, time_scale=time_scale)
y_sk = bmb.get_y_sklearn(status, t_event2)
y_sk_coh, x_sk_coh, w_coh= bmb.get_case_cohort(y_sk, x_mat, prop = 1)
coh_y, coh_x, coh_w, coh_coords = bmb.surv_pre_train(y_sk_coh, x_sk_coh, w_coh)
x_tst, tst_coords = bmb.get_posterior_test(np.unique(y_sk_coh["Survival_in_days"]), x_sk_coh)

# COMMAND ----------

# MAGIC %md
# MAGIC Below the bart model is initiated. There are two parameter objects that needs to be created; the model_dict and sampler_dict.
# MAGIC - model_dict 
# MAGIC     1. trees is the number of trees to generate (20-100 seems to be appropriate). Greater number of trees slows down the analyses.
# MAGIC     2. split_rules is a list of split rules corresponding to each variable in the covariate matrix. The first split rule should always be ContinuousSplitRule since it corresponds to the time column of the long-form x_mat. OneHotSplitRule is used for categorical and binary data, while Continuous should be used for all other variables.
# MAGIC     3. Split prior is a new feature I am working out. It provides the probability of the feature to be selected for a tree. By default this probability is plit uniformly across all variables, but intuition suggests that having a greater probability on the time variable allows for fewer trees as the time is always going to be the major predictor we want included. The split_prior component can be removed if unwanted.
# MAGIC - sampler_dict contains parameters specific for the the sampling step of MCMC. These can all be adjusted. Ideally the number of draws and tune steps should be a balance of sufficiency and computational time. The BART algorithm seems to converge to the sampling distribution quickly, so a smaller number of tune and draws could be beneficial for sampling time.
# MAGIC
# MAGIC The bart class object is instantiated with the model parameters. Internally is constructs the framework for the bart model that can be used in a fit step completed after it is instantiated.
# MAGIC
# MAGIC

# COMMAND ----------

# intitialize models
model_dict = {"trees": 20,
    "split_rules": [
    "pmb.ContinuousSplitRule()",
    "pmb.OneHotSplitRule",
    "pmb.ContinuousSplitRule()",
    "pmb.ContinuousSplitRule()",
    "pmb.ContinuousSplitRule()",
    "pmb.OneHotSplitRule()",
    "pmb.ContinuousSplitRule()",
    "pmb.ContinuousSplitRule()",
    "pmb.ContinuousSplitRule()",
    "pmb.ContinuousSplitRule()",
    "pmb.ContinuousSplitRule()"
    ],
    "split_prior": [0.9,
                0.002,
                0.002,
                0.002,
                0.002,
                0.002,
                0.002,
                0.002,
                0.002,
                0.002,
                0.002]
}

sampler_dict = {
            "draws": 100,
            "tune": 100,
            "cores": 4,
            "chains": 4,
            "compute_convergence_checks": False
        }

# initialize bart
bart_model = bmb.BartSurvModel(model_config=model_dict, sampler_config=sampler_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC Below the model is fit to the data and posterior predictions of the "mu" parameter (this is our probability risk of event occuring at time) is drawn for each obs at each time point. 
# MAGIC
# MAGIC When the model is fit in the same instance that posterior predictions are made we can use the class method sample_posterior_predictive. 
# MAGIC
# MAGIC When a trained model is loaded from a prior instance the bart_predict function is used. This comes from an instability of saving the model object internal to the pymc-bart implementation.
# MAGIC
# MAGIC The class method save will save both the trace of the trained model (idata) and the trained tree structure. In the survival setting only the tree structure is required for future predictions.
# MAGIC
# MAGIC The functions get_prob and get_survival return the predicted risk probabilites and computed survival estimates for each observations for each of the time points. 
# MAGIC - The returned estimates are point-estimates for each patient and stratified credible intervals and point estimates can be drawn from the predictions. These functions will be adjusted to return the raw draws from posterior to compute quantile estimates as well in the future. 

# COMMAND ----------

# fit model
bart_model.fit(coh_y, coh_x, coh_w, coh_coords)
# sample posterior
post = bart_model.sample_posterior_predictive(x_tst, tst_coords, extend_idata=True)
# get posterior data
prob = bmb.get_prob(post)
sv = bmb.get_survival(post)

bart_model.save(idata_name="/tmp/test_idata1.pkl", all_tree_name="/tmp/test_tree1.pkl")

# COMMAND ----------

# save idata and tree
bart_model.save(idata_name="/tmp/test_idata1.pkl", all_tree_name="/tmp/test_tree1.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC A Variable importance can be computed from the trained model. The pymc bart package provides an algortihm that computes additives importance of ranked variable as shown in the blocked out code below. This methods takes time to complete and does not provide a large benefit in comparision to the more naive measure of variable importance as shown in the second code block below.

# COMMAND ----------

# not needed, used simple var-importance
# pmb.plot_variable_importance(
#     idata=bart_model.idata,
#     bartrv=bart_model.model.f,
#     X=x_tst,
#     samples=100
# )

# COMMAND ----------

# MAGIC %md
# MAGIC Naive variable importances shows variable 0 is the most important (the time), followed by several other variables that are included in the data generating process.
# MAGIC
# MAGIC The value associated with each variable is the mean inclusion frequency over all of the draws. Unfortunately, this method is not quantitatively robust and the values provide no quantitative purpose beyond relative rank in assessment in the trained model. 
# MAGIC
# MAGIC It is useful as a overview of importance similar to variable importance measures returned from Random Forest methods and can direct the follow-up with the Marginal Dependence Evaluation.

# COMMAND ----------

# naive variable importance
vars_tree = bart_model.idata.sample_stats.variable_inclusion.values.reshape(400,-1)
vmean = vars_tree.mean(0)
var_dict = dict(zip(np.argsort(-vmean), -np.sort(-vmean)))
var_dict
# save this

# COMMAND ----------

# MAGIC %md
# MAGIC # Cox Model

# COMMAND ----------

# MAGIC %md
# MAGIC Train a cox model for comparison purposes.
# MAGIC
# MAGIC While the exp(coef) returned do not have associated confidence intervals, the point estimate can be used as an approximate comparison of variable importance between cox and bart models.

# COMMAND ----------

# cox model
cph_coef, cph_sv, cph_chz = ut.get_cph(y_sk, x_mat, x_mat)

# COMMAND ----------

cph_coef

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluations

# COMMAND ----------

# MAGIC %md
# MAGIC Compare the estimated survival curves of the two models to the kpm estimator. 

# COMMAND ----------

ut.quick_kpm_plot(y_sk, msk=x_mat[:,0]==1, cph_sv=cph_sv, sv=sv)

# COMMAND ----------

# MAGIC %md
# MAGIC Get evaluations metrics including the cindex and brier score.

# COMMAND ----------

bart_met = ut.get_sv_metrics(sv, prob, y_sk_coh)
cph_met = ut.get_sv_metrics(cph_sv, cph_chz, y_sk)

# log these
print(bart_met)
print(cph_met)

# COMMAND ----------

# MAGIC %md
# MAGIC # Test

# COMMAND ----------

# MAGIC %md
# MAGIC To preform out of sample evaluations use the generated test dataset. 
# MAGIC
# MAGIC The same data preparations can be used as above.
# MAGIC
# MAGIC The bart_model.sample_posterior_predictive can be used to generate predicitons if the testing is be completed in the same instance of the model.fit run. 

# COMMAND ----------

# Test
t_t_event2 = bmb.get_time_transform(t_t_event, time_scale=time_scale)
t_y_sk = bmb.get_y_sklearn(t_status, t_t_event2)
t_y_sk_coh, t_x_sk_coh, t_w_coh= bmb.get_case_cohort(t_y_sk, t_x_mat, 1)
t_x_tst, t_tst_coords = bmb.get_posterior_test(np.unique(t_y_sk_coh["Survival_in_days"]), t_x_sk_coh)

t_post = bart_model.sample_posterior_predictive(t_x_tst, t_tst_coords, extend_idata=False)
t_prob = bmb.get_prob(t_post)
t_sv = bmb.get_survival(t_post)

# cox
t_cph_coef, t_cph_sv, t_cph_chz = ut.get_cph(y_sk, x_mat, t_x_mat)


# COMMAND ----------

# MAGIC %md
# MAGIC Test dataset evaluation metrics.

# COMMAND ----------

ut.quick_kpm_plot(t_y_sk, t_x_mat[:,0]==1, t_cph_sv, t_sv)
# save this
t_bart_met = ut.get_sv_metrics(t_sv, t_prob, t_y_sk_coh, y_sk_coh)
t_cph_met = ut.get_sv_metrics(t_cph_sv, t_cph_chz, t_y_sk, y_sk)

print(t_bart_met)
print(t_cph_met)

# COMMAND ----------

# MAGIC %md
# MAGIC # True Evaluation 

# COMMAND ----------

# MAGIC %md
# MAGIC If working with the simulated data, a evaluation can be completed with the true sv/hz estimates. The primary values estimated include rmse and bias.

# COMMAND ----------

tmp_times = [0,1,2,3,4,5]
b_true_eval = ut.get_true_rmse_bias(
    true_scale["sv_true"][:,1:], 
    sv,
    tmp_times
    )

cph_true_eval = ut.get_true_rmse_bias(
    true_scale["sv_true"][:,1:], 
    cph_sv,
    tmp_times
    )


t_b_true_eval = ut.get_true_rmse_bias(
    t_true_scale["sv_true"][:,1:], 
    t_sv,
    tmp_times
    )

t_cph_true_eval = ut.get_true_rmse_bias(
    t_true_scale["sv_true"][:,1:], 
    t_cph_sv,
    tmp_times
    )

print(b_true_eval)
print(cph_true_eval)
print(t_b_true_eval)
print(t_cph_true_eval)

# COMMAND ----------

# MAGIC %md
# MAGIC A sample comparison of predictions from the different models in comparison to the true sv can also demonstrated through sv plots.

# COMMAND ----------

tmp_times = [1,2,3,4,5,6]

for i in np.arange(0,10):
    if i == 0:
        plt.plot(tmp_times, t_true_scale["sv_true"][i,1:], color= "red", alpha= 0.2, label="exp")
        plt.plot(tmp_times, t_sv[i,:], color= "blue", alpha= 0.2, label="pred_b")
        plt.plot(tmp_times, t_cph_sv[i,:], color= "green", alpha= 0.2, label = "pred_cph")
    else:
        plt.plot(tmp_times, t_true_scale["sv_true"][i,1:], color= "red", alpha= 0.2)
        plt.plot(tmp_times, t_sv[i,:], color= "blue", alpha= 0.2)
        plt.plot(tmp_times, t_cph_sv[i,:], color= "green", alpha= 0.2)

plt.legend()
# t_true_scale
# t_sv[0:100]

# COMMAND ----------

# MAGIC %md
# MAGIC The stratified estimates by the main variable can be plotted to compare how well the model does at computing the sub-group sv means. 

# COMMAND ----------

msk = t_x_mat[:,0] == 1

m1 = t_true_scale["sv_true"][msk,1:].mean(axis=0)
m1_ = t_true_scale["sv_true"][~msk,1:].mean(axis=0)
m2 = t_sv[msk,:].mean(axis=0)
m2_ = t_sv[~msk,:].mean(axis=0)
m3 = t_cph_sv[msk,:].mean(axis=0)
m3_ = t_cph_sv[~msk,:].mean(axis=0)


tmp_times = [1,2,3,4,5,6]
plt.plot(tmp_times, m1, color = "red", label = "exp")
plt.plot(tmp_times, m1_, color= "pink", label = "exp")
plt.plot(tmp_times, m2, color = "blue", label="bart")
plt.plot(tmp_times, m2_, color = "lightblue", label= "bart")
plt.plot(tmp_times, m3, color = "green", label= "cph")
plt.plot(tmp_times, m3_, color = "lightgreen", label = "cph")

plt.legend()


# COMMAND ----------

# MAGIC %md
# MAGIC # PDP Evaluations

# COMMAND ----------

# MAGIC %md
# MAGIC PDP evaluations allow estimations of marginal effects of a variable. This is part of the primary outcome of the proposed study.

# COMMAND ----------

# MAGIC %md
# MAGIC In the databricks workflows this process will be seperated from the model training, to allow for specific pdps to tested. 
# MAGIC
# MAGIC Doing this in a new compute instance requires predictions to be drawn with a different method.
# MAGIC
# MAGIC 1. Instantiate a new surv_bart model and load the saved trace and tree structure.
# MAGIC 2. Collect the training data loaded with the model. 
# MAGIC 3. get_pdp is a function that generates a pdp testing dataset. One or two variables can be selected for the pdp and a subsample of the training data can be used if the data is large.
# MAGIC 4. use get_posterior_test to create the long-form dataset for predictions.

# COMMAND ----------

bart_m2 = bmb.BartSurvModel(model_config=model_dict).load("test_idata1.pkl", "test_tree1.pkl")
x2 = bart_m2.X[bart_m2.X[:,0]==1][:,1:]
pdp_sk, pdp_idx = bmb.get_pdp(x2, var_col = [0], values = [[0,1]], sample_n=1000)
pdp_x, pdp_coords = bmb.get_posterior_test(bart_m2.uniq_times, pdp_sk)

# COMMAND ----------

# MAGIC %md
# MAGIC 1. bart_predict - returns posterior draws of the risk probability, similar to get_posterior_predictions above
# MAGIC 2. get_survival/get_prob to return patientwise point estimates

# COMMAND ----------

pdp_post = bart_m2.bart_predict(pdp_x, pdp_coords)
pdp_sv = bmb.get_survival(pdp_post)
pdp_prob = bmb.get_prob(pdp_post)

# COMMAND ----------

# MAGIC %md
# MAGIC For comparison agains the true in a simulated scenario the sim_surv function can be used with the same parameters used in the initial instance, except the x_mat dataset is replaced with the pdp_sk dataset.

# COMMAND ----------

pdp_scale = ut.sim_surv(pdp_sk, LAMBDA, ALPHA_F, eos, time_scale=time_scale, return_full=False, true_only=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Plot the expected (true) vs predicted values for each patient. The will ideally follow the 1:1 axis.

# COMMAND ----------

# true plot
plt.plot(pdp_scale["sv_true"][:,1], pdp_sv[:,0], "bo", alpha=0.02)
plt.plot(pdp_scale["sv_true"][:,2], pdp_sv[:,1], "ro", alpha=0.02)
plt.plot(pdp_scale["sv_true"][:,3], pdp_sv[:,2], "go", alpha=0.02)
plt.plot(pdp_scale["sv_true"][:,4], pdp_sv[:,3], "ko", alpha=0.02)
plt.plot(pdp_scale["sv_true"][:,5], pdp_sv[:,4], "mo", alpha=0.02)
plt.plot(pdp_scale["sv_true"][:,6], pdp_sv[:,5], "wo", alpha=0.02)

plt.grid(visible=True)
plt.xlim(0,1)
plt.ylim(0,1)
plt.ylabel("pred")
plt.xlabel("exp")

# COMMAND ----------

# MAGIC %md
# MAGIC Plot the pdp comparison of expected pdp and predicted pdp.

# COMMAND ----------

exp0 = pdp_scale["sv_true"][pdp_idx["coord"]==0]
exp1 = pdp_scale["sv_true"][pdp_idx["coord"]==1]
edm = (exp1-exp0).mean(0)
edq = edm - np.quantile((exp1-exp0), [0.025, 0.975], axis=0)

pred0 = pdp_sv[pdp_idx["coord"]==0]
pred1 = pdp_sv[pdp_idx["coord"]==1]
pdm = (pred1-pred0).mean(0)
pdq = pdm - np.quantile((pred1-pred0), [0.025, 0.975], axis=0)

plt.errorbar([1,2,3,4,5,6], y = edm[1:], yerr = np.abs(edq[:,1:]), alpha=0.2, label = "exp")
plt.errorbar([1,2,3,4,5,6], y = pdm, yerr = np.abs(pdq), alpha=0.2, label = "pred")
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC Hazard ratios/Risk ratios can also be computed from the pdp predictions using the risk probabilities.

# COMMAND ----------

# hr
hr_true = (pdp_scale["hz_true"][pdp_idx["coord"]==1]/pdp_scale["hz_true"][pdp_idx["coord"]==0])
hr_true_m = hr_true.mean(0)
hr_true_q = np.quantile(hr_true, [0.025, 0.975], 0)

rr_pred = (pdp_prob[pdp_idx["coord"]==1]/pdp_prob[pdp_idx["coord"]==0])
rr_pred_m = rr_pred.mean(0)
rr_pred_q = np.quantile(rr_pred, [0.025,0.975],0)

# COMMAND ----------

# MAGIC %md
# MAGIC There may be some notable differences in this ratio since the true would be more equivalent to a HR and the expected is a Risk Ratio and some difference exist between these measures.

# COMMAND ----------

print(hr_true_m)
print(hr_true_q)
print(hr_true_m[1:].mean())
print(rr_pred_m)
print(rr_pred_q)
print(rr_pred_m.mean())


# COMMAND ----------

# MAGIC %md
# MAGIC ## General PDP

# COMMAND ----------

# MAGIC %md
# MAGIC Outside of the simulation scenario, each variable can be evaluated with a pdp to derive estimated marginal effect of the variable.
# MAGIC
# MAGIC The magnitude of estimated effects should follow the rank of the variable importance given above.
# MAGIC
# MAGIC NOTE: Sometimes we refer to the variable 0 as the time and othertimes we refer to it as the first variable of the covariate matrix. When generating the PDPs the first variables of covariate matrix is index 0 and once the long form predicton matrix is generated the index 0 variable is now time, moving the first variable to index 1.

# COMMAND ----------

var_pdp = 0
v2q = np.quantile(x2[:,var_pdp], [0.25,0.75])
pdp_sk, pdp_idx = bmb.get_pdp(x2, var_col = [var_pdp], values = [v2q], sample_n=1000)
pdp_x, pdp_coords = bmb.get_posterior_test(bart_m2.uniq_times, pdp_sk)
pdp_post = bart_m2.bart_predict(pdp_x, pdp_coords)
pdp_sv = bmb.get_survival(pdp_post)
pdp_prob = bmb.get_prob(pdp_post)

pred0 = pdp_sv[pdp_idx["coord"]==0]
pred1 = pdp_sv[pdp_idx["coord"]==1]
pdm = (pred1-pred0).mean(0)
pdq = pdm - np.quantile((pred1-pred0), [0.025, 0.975], axis=0)

plt.errorbar([1,2,3,4,5,6], y = pdm, yerr = np.abs(pdq), alpha=0.2, label = "pred")
plt.legend()

rr_pred = (pdp_prob[pdp_idx["coord"]==1]/pdp_prob[pdp_idx["coord"]==0])
rr_pred_m = rr_pred.mean(0)
rr_pred_q = np.quantile(rr_pred, [0.025,0.975], 0)

print(rr_pred_m)
print(rr_pred_q)
print(rr_pred_m.mean())
print(rr_pred_q.mean(1))

# COMMAND ----------

var_pdp = 1
v2q = np.quantile(x2[:,var_pdp], [0.25,0.75])
pdp_sk, pdp_idx = bmb.get_pdp(x2, var_col = [var_pdp], values = [v2q], sample_n=1000)
pdp_x, pdp_coords = bmb.get_posterior_test(bart_m2.uniq_times, pdp_sk)
pdp_post = bart_m2.bart_predict(pdp_x, pdp_coords)
pdp_sv = bmb.get_survival(pdp_post)
pdp_prob = bmb.get_prob(pdp_post)

pred0 = pdp_sv[pdp_idx["coord"]==0]
pred1 = pdp_sv[pdp_idx["coord"]==1]
pdm = (pred1-pred0).mean(0)
pdq = pdm - np.quantile((pred1-pred0), [0.025, 0.975], axis=0)

plt.errorbar([1,2,3,4,5,6], y = pdm, yerr = np.abs(pdq), alpha=0.2, label = "pred")
plt.legend()

rr_pred = (pdp_prob[pdp_idx["coord"]==1]/pdp_prob[pdp_idx["coord"]==0])
rr_pred_m = rr_pred.mean(0)
rr_pred_q = np.quantile(rr_pred, [0.025,0.975], 0)

print(rr_pred_m)
print(rr_pred_q)
print(rr_pred_m.mean())
print(rr_pred_q.mean(1))

# COMMAND ----------

var_pdp = 2
v2q = np.quantile(x2[:,var_pdp], [0.25,0.75])
pdp_sk, pdp_idx = bmb.get_pdp(x2, var_col = [var_pdp], values = [v2q], sample_n=1000)
pdp_x, pdp_coords = bmb.get_posterior_test(bart_m2.uniq_times, pdp_sk)
pdp_post = bart_m2.bart_predict(pdp_x, pdp_coords)
pdp_sv = bmb.get_survival(pdp_post)
pdp_prob = bmb.get_prob(pdp_post)

pred0 = pdp_sv[pdp_idx["coord"]==0]
pred1 = pdp_sv[pdp_idx["coord"]==1]
pdm = (pred1-pred0).mean(0)
pdq = pdm - np.quantile((pred1-pred0), [0.025, 0.975], axis=0)

plt.errorbar([1,2,3,4,5,6], y = pdm, yerr = np.abs(pdq), alpha=0.2, label = "pred")
plt.legend()

rr_pred = (pdp_prob[pdp_idx["coord"]==1]/pdp_prob[pdp_idx["coord"]==0])
rr_pred_m = rr_pred.mean(0)
rr_pred_q = np.quantile(rr_pred, [0.025,0.975], 0)

print(rr_pred_m)
print(rr_pred_q)
print(rr_pred_m.mean())
print(rr_pred_q.mean(1))

# COMMAND ----------

var_pdp = 3
v2q = np.quantile(x2[:,var_pdp], [0.25,0.75])
pdp_sk, pdp_idx = bmb.get_pdp(x2, var_col = [var_pdp], values = [v2q], sample_n=1000)
pdp_x, pdp_coords = bmb.get_posterior_test(bart_m2.uniq_times, pdp_sk)
pdp_post = bart_m2.bart_predict(pdp_x, pdp_coords)
pdp_sv = bmb.get_survival(pdp_post)
pdp_prob = bmb.get_prob(pdp_post)

pred0 = pdp_sv[pdp_idx["coord"]==0]
pred1 = pdp_sv[pdp_idx["coord"]==1]
pdm = (pred1-pred0).mean(0)
pdq = pdm - np.quantile((pred1-pred0), [0.025, 0.975], axis=0)

plt.errorbar([1,2,3,4,5,6], y = pdm, yerr = np.abs(pdq), alpha=0.2, label = "pred")
plt.legend()

rr_pred = (pdp_prob[pdp_idx["coord"]==1]/pdp_prob[pdp_idx["coord"]==0])
rr_pred_m = rr_pred.mean(0)
rr_pred_q = np.quantile(rr_pred, [0.025,0.975], 0)

print(rr_pred_m)
print(rr_pred_q)
print(rr_pred_m.mean())
print(rr_pred_q.mean(1))

# COMMAND ----------

var_pdp = 4
v2q = np.quantile(x2[:,var_pdp], [0.25,0.75])
pdp_sk, pdp_idx = bmb.get_pdp(x2, var_col = [var_pdp], values = [v2q], sample_n=1000)
pdp_x, pdp_coords = bmb.get_posterior_test(bart_m2.uniq_times, pdp_sk)
pdp_post = bart_m2.bart_predict(pdp_x, pdp_coords)
pdp_sv = bmb.get_survival(pdp_post)
pdp_prob = bmb.get_prob(pdp_post)

pred0 = pdp_sv[pdp_idx["coord"]==0]
pred1 = pdp_sv[pdp_idx["coord"]==1]
pdm = (pred1-pred0).mean(0)
pdq = pdm - np.quantile((pred1-pred0), [0.025, 0.975], axis=0)

plt.errorbar([1,2,3,4,5,6], y = pdm, yerr = np.abs(pdq), alpha=0.2, label = "pred")
plt.legend()

rr_pred = (pdp_prob[pdp_idx["coord"]==1]/pdp_prob[pdp_idx["coord"]==0])
rr_pred_m = rr_pred.mean(0)
rr_pred_q = np.quantile(rr_pred, [0.025,0.975], 0)

print(rr_pred_m)
print(rr_pred_q)
print(rr_pred_m.mean())
print(rr_pred_q.mean(1))

# COMMAND ----------

var_pdp = 5
v2q = np.quantile(x2[:,var_pdp], [0.25,0.75])
pdp_sk, pdp_idx = bmb.get_pdp(x2, var_col = [var_pdp], values = [v2q], sample_n=1000)
pdp_x, pdp_coords = bmb.get_posterior_test(bart_m2.uniq_times, pdp_sk)
pdp_post = bart_m2.bart_predict(pdp_x, pdp_coords)
pdp_sv = bmb.get_survival(pdp_post)
pdp_prob = bmb.get_prob(pdp_post)

pred0 = pdp_sv[pdp_idx["coord"]==0]
pred1 = pdp_sv[pdp_idx["coord"]==1]
pdm = (pred1-pred0).mean(0)
pdq = pdm - np.quantile((pred1-pred0), [0.025, 0.975], axis=0)

plt.errorbar([1,2,3,4,5,6], y = pdm, yerr = np.abs(pdq), alpha=0.2, label = "pred")
plt.legend()

rr_pred = (pdp_prob[pdp_idx["coord"]==1]/pdp_prob[pdp_idx["coord"]==0])
rr_pred_m = rr_pred.mean(0)
rr_pred_q = np.quantile(rr_pred, [0.025,0.975], 0)

print(rr_pred_m)
print(rr_pred_q)
print(rr_pred_m.mean())
print(rr_pred_q.mean(1))

# COMMAND ----------

var_pdp = 6
v2q = np.quantile(x2[:,var_pdp], [0.25,0.75])
pdp_sk, pdp_idx = bmb.get_pdp(x2, var_col = [var_pdp], values = [v2q], sample_n=1000)
pdp_x, pdp_coords = bmb.get_posterior_test(bart_m2.uniq_times, pdp_sk)
pdp_post = bart_m2.bart_predict(pdp_x, pdp_coords)
pdp_sv = bmb.get_survival(pdp_post)
pdp_prob = bmb.get_prob(pdp_post)

pred0 = pdp_sv[pdp_idx["coord"]==0]
pred1 = pdp_sv[pdp_idx["coord"]==1]
pdm = (pred1-pred0).mean(0)
pdq = pdm - np.quantile((pred1-pred0), [0.025, 0.975], axis=0)

plt.errorbar([1,2,3,4,5,6], y = pdm, yerr = np.abs(pdq), alpha=0.2, label = "pred")
plt.legend()

rr_pred = (pdp_prob[pdp_idx["coord"]==1]/pdp_prob[pdp_idx["coord"]==0])
rr_pred_m = rr_pred.mean(0)
rr_pred_q = np.quantile(rr_pred, [0.025,0.975], 0)

print(rr_pred_m)
print(rr_pred_q)
print(rr_pred_m.mean())
print(rr_pred_q.mean(1))

# COMMAND ----------

var_pdp = 7
v2q = np.quantile(x2[:,var_pdp], [0.25,0.75])
pdp_sk, pdp_idx = bmb.get_pdp(x2, var_col = [var_pdp], values = [v2q], sample_n=1000)
pdp_x, pdp_coords = bmb.get_posterior_test(bart_m2.uniq_times, pdp_sk)
pdp_post = bart_m2.bart_predict(pdp_x, pdp_coords)
pdp_sv = bmb.get_survival(pdp_post)
pdp_prob = bmb.get_prob(pdp_post)

pred0 = pdp_sv[pdp_idx["coord"]==0]
pred1 = pdp_sv[pdp_idx["coord"]==1]
pdm = (pred1-pred0).mean(0)
pdq = pdm - np.quantile((pred1-pred0), [0.025, 0.975], axis=0)

plt.errorbar([1,2,3,4,5,6], y = pdm, yerr = np.abs(pdq), alpha=0.2, label = "pred")
plt.legend()

rr_pred = (pdp_prob[pdp_idx["coord"]==1]/pdp_prob[pdp_idx["coord"]==0])
rr_pred_m = rr_pred.mean(0)
rr_pred_q = np.quantile(rr_pred, [0.025,0.975], 0)

print(rr_pred_m)
print(rr_pred_q)
print(rr_pred_m.mean())
print(rr_pred_q.mean(1))

# COMMAND ----------

var_pdp = 8
v2q = np.quantile(x2[:,var_pdp], [0.25,0.75])
pdp_sk, pdp_idx = bmb.get_pdp(x2, var_col = [var_pdp], values = [v2q], sample_n=1000)
pdp_x, pdp_coords = bmb.get_posterior_test(bart_m2.uniq_times, pdp_sk)
pdp_post = bart_m2.bart_predict(pdp_x, pdp_coords)
pdp_sv = bmb.get_survival(pdp_post)
pdp_prob = bmb.get_prob(pdp_post)

pred0 = pdp_sv[pdp_idx["coord"]==0]
pred1 = pdp_sv[pdp_idx["coord"]==1]
pdm = (pred1-pred0).mean(0)
pdq = pdm - np.quantile((pred1-pred0), [0.025, 0.975], axis=0)

plt.errorbar([1,2,3,4,5,6], y = pdm, yerr = np.abs(pdq), alpha=0.2, label = "pred")
plt.legend()

rr_pred = (pdp_prob[pdp_idx["coord"]==1]/pdp_prob[pdp_idx["coord"]==0])
rr_pred_m = rr_pred.mean(0)
rr_pred_q = np.quantile(rr_pred, [0.025,0.975], 0)

print(rr_pred_m)
print(rr_pred_q)
print(rr_pred_m.mean())
print(rr_pred_q.mean(1))

# COMMAND ----------

var_pdp = 9
v2q = np.quantile(x2[:,var_pdp], [0.25,0.75])
pdp_sk, pdp_idx = bmb.get_pdp(x2, var_col = [var_pdp], values = [v2q], sample_n=1000)
pdp_x, pdp_coords = bmb.get_posterior_test(bart_m2.uniq_times, pdp_sk)
pdp_post = bart_m2.bart_predict(pdp_x, pdp_coords)
pdp_sv = bmb.get_survival(pdp_post)
pdp_prob = bmb.get_prob(pdp_post)

pred0 = pdp_sv[pdp_idx["coord"]==0]
pred1 = pdp_sv[pdp_idx["coord"]==1]
pdm = (pred1-pred0).mean(0)
pdq = pdm - np.quantile((pred1-pred0), [0.025, 0.975], axis=0)

plt.errorbar([1,2,3,4,5,6], y = pdm, yerr = np.abs(pdq), alpha=0.2, label = "pred")
plt.legend()

rr_pred = (pdp_prob[pdp_idx["coord"]==1]/pdp_prob[pdp_idx["coord"]==0])
rr_pred_m = rr_pred.mean(0)
rr_pred_q = np.quantile(rr_pred, [0.025,0.975], 0)

print(rr_pred_m)
print(rr_pred_q)
print(rr_pred_m.mean())
print(rr_pred_q.mean(1))

# COMMAND ----------

# MAGIC %md
# MAGIC Comparisons of the estimated Risk Ratio can be compared to CPH HR that are listed below.

# COMMAND ----------

cph_coef
