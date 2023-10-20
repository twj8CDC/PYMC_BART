
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

#################################

library(BART)

B <- getOption('mc.cores', 1)
figures = getOption('figures', default='NONE')

## load survival package for the advanced lung cancer example
data(lung)

N <- length(lung$status)
N
table(lung$ph.karno, lung$pat.karno)

## if physician's KPS unavailable, then use the patient's
h <- which(is.na(lung$ph.karno))
lung$ph.karno[h] <- lung$pat.karno[h]

times <- lung$time
delta <- lung$status-1 ##lung$status: 1=censored, 2=dead
##delta: 0=censored, 1=dead

## this study reports time in days rather than weeks or months
## coarsening from days to weeks or months will reduce the computational burden
##times <- ceiling(times/30)
times <- ceiling(times/30)  ## weeks

##table(times)
table(delta)

## matrix of observed covariates
x.train <- cbind(lung$sex, lung$age, lung$ph.karno)
x.train
## lung$sex:        Male=1 Female=2
## lung$age:        Age in years
## lung$ph.karno:   Karnofsky performance score (dead=0:normal=100:by=10)
##                  rated by physician

dimnames(x.train)[[2]] <- c('M(1):F(2)', 'age(39:82)', 'ph.karno(50:100:10)')

table(x.train[ , 1])
summary(x.train[ , 2])
table(x.train[ , 3])

## run one long MCMC chain in one process
## set.seed(99)
## post <- surv.bart(x.train=x.train, times=times, delta=delta, x.test=x.test)

## in the interest of time, consider speeding it up by parallel processing
## run "mc.cores" number of shorter MCMC chains in parallel processes
post <- mc.surv.bart(x.train=x.train, times=times, delta=delta,
                     mc.cores=B, seed=99, K=100)

pre <- surv.pre.bart(times=times, delta=delta, x.train=x.train,
                     x.test=x.train, K=100)

str(post)
K <- pre$K
M <- post$ndpost
NK <- N*K

pre$tx.test <- rbind(pre$tx.test, pre$tx.test)
pre$tx.test[ , 2] <- c(rep(1, N*K), rep(2, N*K))
pre$tx.test[ , 2] <- rep(1:2, each=NK)
## sex pushed to col 2, since time is always in col 1

pred <- predict(post, newdata=pre$tx.test, mc.cores=B)

for(i in seq(1, N, by=5)) {
##for(i in 1:N) {
    h=(i-1)*K+1:K
    if(i==1)
        plot(c(0, pre$times), c(1, pred$surv.test.mean[h]),
             type='s', col=4, lty=2,
             ylim=0:1, ylab='S(t, x)', xlab='t (weeks)',)
    else lines(c(0, pre$times), c(1, pred$surv.test.mean[h]),
               type='s', col=4, lty=2)
    lines(c(0, pre$times), c(1, pred$surv.test.mean[h+NK]),
          type='s', col=2, lty=3)
}

if(figures!='NONE')
    dev.copy2pdf(file=paste(figures, 'lung-ice.pdf', sep='/'))


# The ice plot is just the individual conditional plots (non averaged)
# if we wanted to evaluate another condition we can add a coloration on the subgroup

##########################################
# RElative Risk
RR = matrix(nrow=post$ndpost, ncol=K)

# RR matrix is draws * times
for(j in 1:K) {
    h=seq(j, NK, K)
    RR[ , j]=apply(pred$prob.test[ , NK+h]/pred$prob.test[ , h], 1, mean)
}
str(RR)

# mean RR averaged over draws
RR.mean = apply(RR, 2, mean)
RR.lower = apply(RR, 2, quantile, 0.025)
RR.upper = apply(RR, 2, quantile, 0.975)
RR.mean

# averaged RR over times
RR.mean. = apply(RR, 1, mean)
RR.lower. = quantile(RR.mean., probs=0.025)
RR.upper. = quantile(RR.mean., probs=0.975)
RR.mean. = mean(RR.mean.)
# RR over times w/ prop hazard
RR.mean. 
str(RR.mean.)
RR.lower.
RR.upper.

# par(2,1)
plot(post$times, RR.mean, type='l', log='y')
plot(post$times, RR.mean, col="blue", type="l")

times
data.frame(x.train)
coxph(Surv(times, delta)~., data=data.frame(x.train))

# Relative Risk is the same as the Risk Ratio, Rate Ratio, relative rate
