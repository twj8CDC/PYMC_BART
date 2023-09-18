library(BART)


run = function(input, output) {
    test = read.csv(input)

    # set values
    delta = test$status
    times = test$time

    # get x matrix
    test_names = names(test) 
    x_names = grep("X[0-9]+", test_names)
    x_mat = as.matrix(test[x_names])

    # train 
    post = mc.surv.bart(x.train = x_mat, times=times, delta=delta, x.test = x_mat, mc.cores=8, seed=99)
    
    # get unique times
    tst_times = sort(unique(times))

    # order the x matrix to match python
    tst_x = unique(x_mat)
    sq_x = seq(1, ncol(tst_x),1)
    ord_x = eval(parse(text = paste0('order(', paste0("tst_x[,", sq_x, "]", collapse=","), ")")))
    tst_x = matrix(tst_x[ord_x,])
    
    # set N
    N = length(tst_times)

    # fake matrix is N * nrow by ncol + 1
    out_x = matrix(nrow = N * nrow(tst_x), ncol = ncol(tst_x) + 1)
    names = c("t")
    # create test matrix blocks and assign to the out_x
    for(i in 1:nrow(tst_x)){
        g = matrix(tst_times)
        
        for(j in 1:ncol(tst_x)){
            o = rep(tst_x[i,j], N)
            g = cbind(g, o)
            if(i == 1){
                names = append(names, paste0("x",j))
            }
        }
        # assign
        r1 = 1+((i-1)*N)
        r2 = i*N
        out_x[r1:r2,] = g
    }
    # set names for out_x
    dimnames(out_x)[[2]] = names

    # get predictions
    pred = predict(post, newdata = out_x, mc.cores=8)
    
    # create the final csv to output
    df = cbind(data.frame(post$tx.test), post$prob.test.mean)
    df = cbind(df, post$surv.test.mean)
    names(df) = c(names, "prob", "surv")

    # Cindex=function(risk, times, delta=NULL){   
    #     N=length(risk)
    #     if(N!=length(times))
    #         stop('risk and times must be the same length')
    #     if(length(delta)==0) delta=rep(1, N)
    #     else if(N!=length(delta))
    #         stop('risk and delta must be the same length')

    #     l=0
    #     k=0
    #     for(i in 1:N) {
    #         h=which((times[i]==times & delta[i]>delta) |
    #                 (times[i]<times & delta[i]>0))
    #         if(length(h)>0) {
    #             l=l+sum(risk[i]>risk[h])
    #             k=k+length(h)
    #         }
    #     }
    #     return(l/k)
    # }
    
    K = post$K
    NK = length(post$prob.test.mean)
    seqs = seq(1,NK, K)


    # print(Cindex(post$surv.test.mean[seqs], times, delta))
    # for freidman
    # df = cbind(data.frame(out_x), pred$surv.test.mean)
    # names(df) = c(names, "surv")

    # get id
    lbl_mrg = names(df)[grep("x", names(df))]
    mm = paste0("df$",lbl_mrg, collapse=", ")
    id = eval(parse(text = paste0("paste0(",mm, ")")))
    df["id"] = paste0("i",id)

    write.csv(df, output)
    # return(post)
}


# Cindex=function(risk, times, delta=NULL){   
#     N=length(risk)
#     if(N!=length(times))
#         stop('risk and times must be the same length')
#     if(length(delta)==0) delta=rep(1, N)
#     else if(N!=length(delta))
#         stop('risk and delta must be the same length')

#     l=0
#     k=0
#     for(i in 1:N) {
#         h=which((times[i]==times & delta[i]>delta) |
#                 (times[i]<times & delta[i]>0))
#         if(length(h)>0) {
#             l=l+sum(risk[i]>risk[h])
#             k=k+length(h)
#         }
#     }
#     return(l/k)
# }

args = commandArgs(trailingOnly=TRUE)
try(run(args[1], args[2]))


# args=c("tmp_input.csv", "tmp_output.csv")
# setwd("eval_4_metrics")
# getwd()
# post =run(args[1], args[2])
# train = data.frame("status" = rep(1,10), "time" = c(1,7,6,2,1,2,5,1,9,10), "r" = c(1,0,0,0,1,1,0,1,0,0))
# train
# Cindex(train$r, times=train$time, delta=train$status)
# post$K
# K = post$K
# NK = length(post$prob.test.mean)
# seqs = seq(1,NK, K)
# post$prob.test.mean[seqs]
# post$prob.test.mean
# Cindex(post$surv.test.mean[seq], times=times, delta = delta)