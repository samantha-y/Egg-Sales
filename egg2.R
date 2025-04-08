library(TSA)
library(tseries)
library(ggplot2)
library(dplyr)
library(lubridate)
library(zoo)
getAnywhere(arima)

# https://www.kaggle.com/datasets/kanchana1990/egg-sales-of-a-local-shop-for-30-years/data

all_egg_data <- read.table("train_egg_sales.txt",
                           sep=";")
head(all_egg_data)

# rename columns and drop the first row since it only has column names
all_egg_data <- rename(all_egg_data, 
                       Date = V1,
                       Egg.Sales = V2)
all_egg_data = all_egg_data[-1,]
head(all_egg_data)

typeof(all_egg_data$Date)
typeof(all_egg_data$Egg.Sales)

all_egg_data$Date <- as.Date(all_egg_data$Date)
all_egg_data$Egg.Sales <- as.numeric(all_egg_data$Egg.Sales)

# use a shorter time series for forecasting
# trim to the start of a week
all_egg_data <- subset(all_egg_data,
                       all_egg_data$Date >= as.Date("2010-01-04") & 
                         all_egg_data$Date <= as.Date("2021-12-26"))

plot(all_egg_data$Date,
     all_egg_data$Egg.Sales,
     type="l")

# there are zeros during 2020, presumably due to pandemic
# replace the 0 with interpolation
# otherwise, box-cox won't work
which(all_egg_data$Egg.Sales == 0)
length(which(all_egg_data$Egg.Sales == 0))
all_egg_data$Egg.Sales[all_egg_data$Egg.Sales == 0] <- NA

# Only for the purpose of being able to use Box-Cox, approximate these zero values
# It may not be true to the original data but will probably not affect forecasting
# No more zero values
all_egg_data$Egg.Sales <- na.approx(all_egg_data$Egg.Sales)
plot(all_egg_data$Date,
     all_egg_data$Egg.Sales,
     type="l")

start_date <- min(all_egg_data$Date)
start_year <- as.numeric(format(start_date, "%Y"))
start_day_of_year <- as.numeric(format(start_date, "%j"))

all_egg_data$week <- as.Date(cut(all_egg_data$Date, "week"))
egg_weekly <- aggregate(Egg.Sales ~ week, all_egg_data, sum)

head(egg_weekly)
tail(egg_weekly)

egg <- ts(data=egg_weekly$Egg.Sales,
          start = c(start_year, start_day_of_year),
          frequency = 52)

# figure out the frequency of repetition
spec_rate <- spectrum(egg,method="pgram")
plot(x=1/spec_rate$freq,y=spec_rate$spec,
     main="Spectral Density Estimation For Egg Sales",
     xlab="Period",
     ylab="Spectral density",
     type="l",
     xlim=c(0,10),
     # ylim=c(0,350),
     yaxs="i",
     xaxs="i",
     las=1,
     cex.lab=1.2,
     cex.main=1.2,
     cex.axis=1.2)

max(spec_rate$spec)
rates <- sort(spec_rate$spec, decreasing = TRUE)
idx <- c(which(spec_rate$spec==rates[1]),
         which(spec_rate$spec==rates[2]))
1/spec_rate$freq[idx[1]] # annual, 1.00 year
1/spec_rate$freq[idx[2]] # 0.25 year, quarterly every season)


plot(egg,
     main="Egg Sales Over Time",
     xlab="Time",
     ylab="Egg Sales") # looks cyclic

adf.test(egg) # it is stationary

acf(egg)

egg_lambda <- BoxCox.ar(egg, method="mle")
egg_lambda

lambda <- egg_lambda$mle

egg_trans <- (egg^lambda - 1) / lambda

plot(egg_trans)
acf(egg_trans)
pacf(egg_trans)

egg_fdiff <- diff(egg_trans)
plot(egg_fdiff,
     main="First Differences of Egg Sales",
     xlab="Time",
     ylab="Egg Sales")

acf(c(egg_fdiff),
    main="Autocorrelation - First Differences") # MA(1) but seasonal
pacf(c(egg_fdiff,lag.max=50),
     main="Partial Autocorrelation - First Differences")# no AR, use seasonal model

# try seasonal differencing
egg_sfdiff <- diff(egg_fdiff,lag=52)
plot(egg_sfdiff) 
acf(c(egg_sfdiff),lag.max=50,
    main="Autocorrelation - First Seasonal Differences") # MA(1)
pacf(c(egg_sfdiff),lag.max=50,
     main="Partial Autocorrelation - First Seasonal Differences") # AR?

# we tentatively specify ARIMA(0,1,1) x (1,1,1)_52
egg_mod2 = TSA::arima(egg_trans, order = c(0,1,1), seasonal = list(order = c(1,1,1), period=52))
egg_mod2

# model diagnostics
# residuals
plot(rstandard(egg_mod2),type="l",
     main="Residuals of Model",
     xlab="Residuals")

# QQ plot: pretty much normal
qqnorm(residuals(egg_mod2))
qqline(residuals(egg_mod2))

# test still stationary
shapiro.test(egg_sfdiff)

# other diagnostics: ACF, Ljung-Box test statistic
tsdiag(egg_mod2)


# use model to predict
# plot predictions
plot(x=egg_mod2,
     n.ahead=52,
     main="Egg Sales Prediction",
     ylab="Egg Sales",
     xlab="Time [years]",
     xlim = c(2021,2023),
     n1=c(start_year, start_day_of_year),
     pch=19)
