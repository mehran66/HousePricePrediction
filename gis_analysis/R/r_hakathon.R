data = read.csv("C:/Users/mehra/Desktop/HousePricePrediction/HousePricePrediction/cleaned_house_data.csv")  # Load some data
summary(data)
cor.test(data$price, data$gross_square)  # Correlation test
m = lm(price ~ gross_square, data=data)  # Fit a model
plot(price ~ gross_square, data=data)  # Scatterplot...
abline(m, col="red")  # ...with best fit line


# Converting the values from umeric (integers) to categorical data
data$district = as.factor(data$district)
data$neighborhood = as.factor(data$neighborhood)
data$building_class_category  = as.factor(data$building_class_category)
data$tax_class_at_present = as.factor(data$tax_class_at_present)
data$building_class_at_present = as.factor(data$building_class_at_present)
data$tax_class_at_time_of_sale = as.factor(data$tax_class_at_time_of_sale)
data$tax_building_class_at_time_of_sale = as.factor(data$building_class_at_time_of_sale)

#_______________________________________________________

# distribution
boxplot(data$price ~ data$district,
        col = "beige",
        xlab = "District",
        ylab = "Price", outline =F)

cor.test(data$price, data$year_built)  # Correlation test
m = lm(price ~ year_built, data=data)  # Fit a model
plot(price ~ year_built, data=data)  # Scatterplot...
abline(m, col="red")  # ...with best fit line

#_______________________________________________________



dataSubset = data.frame(District = data$district,Neigborhood = data$neighborhood,
                        BuildingClass = data$building_class_category, TaxClass = data$tax_class_at_present,
                        LandArea = data$land_square_feet, GrossArea = data$gross_square_feet,
                        YearBuilt = data$year_built,Price =data$price)

dataSubset$District = as.numeric(dataSubset$District)
dataSubset$Neigborhood = as.numeric(dataSubset$Neigborhood)
dataSubset$BuildingClass  = as.numeric(dataSubset$BuildingClass)
dataSubset$TaxClass = as.numeric(dataSubset$TaxClass)
dataSubset$LandArea = as.numeric(dataSubset$LandArea)
dataSubset$GrossArea = as.numeric(dataSubset$GrossArea)
dataSubset$YearBuilt = as.numeric(dataSubset$YearBuilt)
dataSubset$Price =as.numeric(dataSubset$Price)

dataSubset = dataSubset(dataSubset$Price <1000000)

dataSubset_corr = cor(dataSubset) # get correlations

library('corrplot') #package corrplot
corrplot(dataSubset_corr, method = "circle", addCoef.col = "black") #plot matrix

#_______________________________________________________
# distribution
boxplot(data$price ~ data$district,
        col = "beige",
        xlab = "District",
        ylab = "Price", outline =F)

boxplot(data$price ~ data$building_class_category,
        col = "beige",
        xlab = "building class category",
        ylab = "Price", outline =F)

boxplot(data$price ~ data$tax_class_at_present,
        col = "beige",
        xlab = "tax class category",
        ylab = "Price", outline =F)

boxplot(data$price ~ data$land_square,
        col = "beige",
        xlab = "Zip code",
        ylab = "Price", outline =F)



p = ggplot(data, aes(land_square, price), outlier.shape = NA) +
  geom_point(color="firebrick") +
  facet_wrap(~district, scales = "free") + geom_smooth(method=lm)+
  xlab("Land area (sq feet)") +
  ylab ("Price")
print(p)
