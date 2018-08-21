##Project: ANA625
##Author: Kornkanok Somkul, Shashi Bala
##Project name: - Telcom 
#import file
Telcom <- read.csv("/Users/jill_jewelry/Documents/MSDA/ANA625/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
#show the structure
str(Telcom)
View(Telcom)

## Check missing values
str(Telcom)
is.na(Telcom) # TRUE for NA values
complete.cases(Telcom) # FALSE for rows that have some NA values
table(is.na(Telcom)) #TRUE for NA values
# there are 11 missing values

# Omit missing values
Telcom2 <- Telcom[complete.cases(Telcom), ]
Telcom2 <- na.omit(Telcom)
table(is.na(Telcom2))
str(Telcom2)
# the obs decrease from 7043 to 7032 (11 obs were deleted).

save(Telcom2, file="Telcom2.RData")

#check balance in dataset
par("mar")
par(mar=c(1,1,1,1))
plot(Telcom2$Churn)

summary(Telcom2)
# In summary output, it shows min, max, median, mean, 1st and 3rd quartiles in each numeric variables. 
# and for categorical variables, it shows counts for each level in each variable.

#interquartile range
IQR(Telcom2$tenure)
IQR(Telcom2$MonthlyCharges)
IQR(Telcom2$TotalCharges)

#Mode
Mode<- function(x) { 
  ux <- unique(x) 
  ux[which.max(tabulate(match(x, ux)))]
}
Mode(Telcom2$tenure)
Mode(Telcom2$MonthlyCharges)
Mode(Telcom2$TotalCharges)

#standard diavation
sd(Telcom2$tenure)
sd(Telcom2$MonthlyCharges)
sd(Telcom2$TotalCharges)

#variance
var(Telcom2$tenure)
var(Telcom2$MonthlyCharges)
var(Telcom2$TotalCharges)

#Skewness and Kurtosis
install.packages("moments")
library(moments)
skewness(Telcom2$tenure)
skewness(Telcom2$MonthlyCharges)
skewness(Telcom2$TotalCharges)
kurtosis(Telcom2$tenure)
kurtosis(Telcom2$MonthlyCharges)
kurtosis(Telcom2$TotalCharges)

#Histrogram
hist(Telcom2$tenure, col= "pink", border = "red", 
     main = "Histogram of length of customer relationship", 
     xlab="tenure")

hist(Telcom2$MonthlyCharges, col="orange", border = "purple", 
     main = "Histogram of Monthly Charges", 
     xlab="MonthlyCharges")

hist(Telcom2$TotalCharges, col="lightblue", border = "blue", 
     main = "Histogram of Total Charges", 
     xlab="TotalCharges")




