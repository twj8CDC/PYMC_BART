
library(BART)

df = read.csv("eval_4.csv")[c("status", "time", "X0")]

class(df[c("time","X0")])
xtrain = as.matrix(df[c("X0")])
delta = as.matrix(df["status"])
times = as.matrix(df["time"])

test = as.matrix(sort(unique(xtrain)))

post = mc.surv.bart(x.train = xtrain, delta=delta, times=times, x.test = xtrain, mc.cores = 8)

str(post)

write.csv(post$surv.test.mean, "eval_4_tmp.csv")

post$times
plot(c(0, post$times), c(1,post$surv.test.mean[1:5]), type="l")
lines(c(0, post$times), c(1,post$surv.test.mean[6:10]), type="l")

# concordance
str(post)
t(matrix(post$prob.test.mean, nrow=5, ncol=2))
K = post$K


# q <- 1-pnorm(post$yhat.test)
# for(j in 1:K) {
#     if(j==1) P <- q[ , K+1]-q[ , 1]
#     else P <- P+(q[ , K+j]-q[ , j])*post$surv.test[ , K+j-1]*post$surv.test[ , j-1]
# }
# C <- 0.5*(1-P)
# summary(C)




# cindex
Cindex=function(risk, times, delta=NULL){   
    N=length(risk)
    if(N!=length(times))
        stop('risk and times must be the same length')
    if(length(delta)==0) delta=rep(1, N)
    else if(N!=length(delta))
        stop('risk and delta must be the same length')

    l=0
    k=0
    for(i in 1:N) {
        h=which((times[i]==times & delta[i]>delta) |
                (times[i]<times & delta[i]>0))
        if(length(h)>0) {
            l=l+sum(risk[i]>risk[h])
            k=k+length(h)
        }
    }
    return(l/k)
}

matrix(post$prob.train.mean, nrow = 5, ncol=100)
# str(post$prob.train.mean)

K
C = 0
NK = length(post$prob.test.mean)
for(j in 1:K){
    print(j)
    C[j] = Cindex(post$prob.test.mean[seq(j,NK,K)], times = times, delta = delta)
}


