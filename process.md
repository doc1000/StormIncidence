

poisson.test(c(n1, n2), c(t1, t2), alternative = c("two.sided"))

confidence intervals: Lrec = as.numeric(as.Date("2010-07-01") - as.Date("2007-12-02")) # Length of recession Lnrec = as.numeric(as.Date("2007-12-01") - as.Date("2001-12-01")) # L of non rec period (43/Lrec)/(50/Lnrec)

N = 100000 k=(rpois(N, 43)/Lrec)/(rpois(N, 50)/Lnrec) c(quantile(k, c(0.025, .25, .5, .75, .975)), mean=mean(k), sd=sd(k))



poisson.ppf(0.95,310)
poisson.ppf(0.05,310)
prob = poisson.cdf(x, mu)
