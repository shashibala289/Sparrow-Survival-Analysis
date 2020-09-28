#set working directory
setwd("C:\\Users\\shashi\\Desktop\\ProjectBAN600")


#read the desired file
sparrow <- readRDS("sparrow.rds")
sparrow.original <-readRDS("sparrow.original.rds")

library(dplyr)
glimpse(sparrow)
library(DataExplorer)
library(ggplot2)
library(sqldf)
library(broom)
library(stargazer)
library(caret)
library(car)

#selecting desired columns (continuous variables) from dataset (sparrow)
data <- data.frame(sparrow$total_length, sparrow$wingspan, sparrow$weight, sparrow$beak_head, sparrow$humerus, sparrow$femur, sparrow$legbone, sparrow$skull, sparrow$sternum)
remove(data)

#removing desired columns from dataset
sparrow$status <- NULL
sparrow$age <- NULL

#exploratory analysis
dim(sparrow)
summary(sparrow)
object.size(sparrow)

#looking for missing values in a dataset
is.na(sparrow)
plot_missing(sparrow)
#scatter plot
attach(sparrow)
plot(total_length, wingspan, main = "Scatterplot", col = "blue")

#histogram
plot_histogram(sparrow)

#variance
var(sparrow$total_length)
var(sparrow$wingspan)
var(sparrow$weight)
var(sparrow$beak_head)
var(sparrow$humerus)
var(sparrow$femur)
var(sparrow$legbone)
var(sparrow$skull)
var(sparrow$sternum)

#standard deviation
sd(sparrow$total_length)
sd(sparrow$wingspan)
sd(sparrow$weight)
sd(sparrow$beak_head)
sd(sparrow$humerus)
sd(sparrow$femur)
sd(sparrow$legbone)
sd(sparrow$skull)
sd(sparrow$sternum)

#correlation 
humerus.wingspan <- subset(sparrow, select = c("humerus", "wingspan"))
cor(humerus.wingspan)
length.wingspan <- subset(sparrow, select = c("total_length", "wingspan"))
cor(length.wingspan)
plot_correlation(sparrow)
plot_correlation(humerus.wingspan)
plot_correlation(length.wingspan)
rm(length.humerus)
rm(length.weight)

#scatterplot for total length and wingspan
library(car)
scatterplot(sparrow$total_length ~ sparrow$wingspan, xlab = "Total Length", ylab = "Wingspan", 
            main ="Scatter Plot for Total-Lenght and Wingspan", labels = row.names(sparrow))

#scatterplot matrices for all variables
pairs(~total_length+wingspan+weight+beak_head+humerus+femur+legbone+skull+sternum, 
      data=sparrow, main="Scatterplot Matrices", col="blue")

#categorical variables
sparrow.original$age = factor(sparrow.original$age,
                              levels = c('adult', 'juvenile'),
                              labels = c(1,0))

sparrow.original$status = factor(sparrow.original$status,
                                 levels = c('Perished','Survived'),
                                 labels = c(FALSE,TRUE))

glm <- glm(status ~ age + total_length + wingspan + weight + beak_head + humerus + femur + 
             legbone + skull + sternum,
           family = 'binomial', data = sparrow.original)
summary(glm)

#regression analysis using only continuous variables 
reg <- lm(wingspan ~ total_length + weight + beak_head + humerus+ femur + legbone + skull + sternum, data = sparrow)
summary(reg)
