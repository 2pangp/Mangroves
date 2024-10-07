# Set work directory.
setwd("C:/Users/e0950361/Desktop/iCloudDrive/Desktop/Mangroves")
setwd("/Users/zhangfengqi/Desktop/Mangroves")

# Import packages.
library(ranger)
library(terra)
library(sp)
library(dplyr)
library(caret)
library(MatchIt)
library(mgcv)
library(ggplot2)
library(ggpubr)
library(MASS)
library(lme4)
library(nlme)
library(DHARMa)
library(tidyverse)
library(gstat)
library(dagitty)
library(ncf)
library(RSpectra)
library(fields)
library(ggeffects)

# DAG ---------------------------------------------------------------------
# Create a DAG based on domain knowledge.
dag <- dagitty('dag {
    PA [pos="0,5"]
    loss [pos="10,5"]
    GDP [pos="1,7"]
    pop [pos="2,8"]
    NRL [pos="7,9"]
    HF [pos="7,7"]
    city [pos="5,7"]
    road [pos="4,6"]
    coast [pos="4,3"]
    river [pos="5,1"]
    crop [pos="9,8"]
    sea_level [pos="7,1"]
    typhoon [pos="9,3"]

    river -> PA -> loss
    river -> loss
    river -> GDP -> PA -> loss
    river -> pop -> PA -> loss
    river -> pop -> GDP -> PA -> loss
    coast -> PA -> loss
    coast -> loss
    coast -> GDP -> PA -> loss
    coast -> pop -> PA -> loss
    coast -> pop -> GDP -> PA -> loss
    
    road -> PA -> loss
    road -> loss
    road -> GDP -> PA -> loss
    road -> pop -> PA -> loss
    road -> pop -> GDP -> PA -> loss
    road -> HF -> loss
    city -> PA -> loss
    city -> loss
    city -> GDP -> PA -> loss
    city -> pop -> PA -> loss
    city -> pop -> GDP -> PA -> loss
    city -> HF -> loss
    
    pop -> HF -> loss
    pop -> HF -> crop -> loss
    pop -> GDP -> HF -> loss
    pop -> GDP -> NRL -> loss
    pop -> GDP -> NRL -> PA -> loss
    pop -> GDP -> crop -> loss
    pop -> crop -> loss
    
    NRL -> loss
    NRL -> PA -> loss
    
    crop -> loss
    
    sea_level -> loss
    typhoon -> loss
}')
plot(dag)
paths(dag, "PA", "loss")
adjustmentSets(dag, "PA", "loss", type = "canonical")

# Random Forest models with latest version of dataset. --------------------
# Import the latest version of dataset.
mangrove <- read.csv("./Data/Mangrove0812.csv") %>% 
  mutate(gdp = ifelse(is.na(gdp), 0, gdp),
         pop = ifelse(is.na(pop), 0, pop),
         hf = ifelse(is.na(hf), 0, hf)) %>% 
  mutate(pop = ifelse(pop < 0, 0, pop)) %>% 
  dplyr::select(-c("X.4", "X.3", "X.2", "X.1", "X", "x", "y")) %>% 
  rename(x = x.1,
         y = y.1) %>% 
  na.omit()
summary(mangrove)
# Calculate the thresholds used for removing outliers.
# IQR_gdp <- IQR(mangrove$gdp, na.rm = T)
# IQR_hf <- IQR(mangrove$hf, na.rm = T)
# IQR_pop <- IQR(mangrove$pop, na.rm = T)
# 
# Q_gdp <- quantile(mangrove$gdp, probs = c(0.25, 0.75), na.rm = T)
# Q_hf <- quantile(mangrove$hf, probs = c(0.25, 0.75), na.rm = T)
# Q_pop <- quantile(mangrove$pop, probs = c(0.25, 0.75), na.rm = T)
# 
# mangrove <- mangrove[ - which((mangrove$gdp > Q_gdp[2] + 5 * IQR_gdp) | (mangrove$gdp < Q_gdp[1] - 5 * IQR_gdp) |
#                     (mangrove$pop > Q_pop[2] + 5 * IQR_pop) | (mangrove$pop < Q_pop[1] - 5 * IQR_pop) ),]
# summary(mangrove)

# Fitting Random Forest model
set.seed(10)
rand_idx <- sample(nrow(mangrove), 0.8*nrow(mangrove))
train_mangrove <- mangrove[rand_idx,]
test_mangrove <- mangrove[-rand_idx,]
mod_f <- as.formula(loss ~ gdp  + hf  + pop  + city + road + PA + coconut + coffee + olp + 
                      rice + rubber + sugarcane + loss_prev + river + coastline + typhoon + sealevel)

rf = ranger(mod_f, data = train_mangrove, num.trees = 500, mtry = 18/3, importance = 'impurity', verbose = T)
rf_pred = predict(rf, data = test_mangrove)
RMSE(rf_pred$predictions, test_mangrove$loss) / (max(test_mangrove$loss) - min(test_mangrove$loss))

rf <- readRDS("./models/RF.rds")


brt <- readRDS("./Models/brtMan.rds")
summary(brt)
brt_pred = predict(brt, newdata = test_mangrove, type = "response")
summary(brt_pred)
RMSE(pred = brt_pred, obs = test_mangrove$loss) / (max(test_mangrove$loss) - min(test_mangrove$loss))

brt_df <- data.frame(obs = test_mangrove$loss, pred = brt_pred)

# Sensitivity analysis for Random Forest model.
Sensitivity_Analysis <- function(model, test_data, test_var)
{
  results <- data.frame()
  seq_values <- seq(min(test_data[[test_var]]), max(test_data[[test_var]]), length.out = 100)
  i <- 1
  for (value in seq_values)
  {
    test_data[[test_var]] <- value
    predictions <- predict(model, data = test_data)$predictions
    mean_prediction <- mean(predictions)
    sd_prediction <- sd(predictions)
    results <- rbind(results, data.frame(value = value, mean_prediction = mean_prediction, sd_prediction = sd_prediction))
    print(paste0(i, " out of 100 iterations: ", value))
    i <- i + 1
  }
  return(results)
}

mod_f <- as.formula(loss ~ gdp  + hf  + pop  + city + road + PA + coconut + coffee + olp + 
                      rice + rubber + sugarcane + loss_prev + river + coastline + typhoon + sealevel)
# Variables require sensitivity analysis.
var_names <- c('hf', 'city', 'road', 'PA', 'river', 'coastline', 'typhoon')
var_desciptions <- c('Human footprint', 'City accessibility', 'Road accessibility', 'Protected Area', 'River accessibility', 'Coastline accessibility', 'Storm frequency')

results_sensitivity <- list()
for (var in var_names)
{
  results_sensitivity[[var]] <- Sensitivity_Analysis(rf, test_mangrove, var)
  print(paste0("Sensitivity analysis for ", var, " is done."))
}
for (var in var_names)
{
  write.csv(results_sensitivity[[var]], paste0("./Data/results_sensitivity_", var, ".csv"), row.names = F)
  print(paste0("Sensitivity analysis for ", var, " is saved."))
}

results_sensitivity <- list()
mod_sensitivity <- list()
significance_sensitivity <- list()
for (i in 1:length(var_names))
{
  results_sensitivity[[var_names[i]]] <- read.csv(paste0("./Data/sensitivity_analysis/results_sensitivity_", var_names[i], ".csv"))
  mod_sensitivity[var_names[[i]]] <- lm(mean_prediction ~ value, data = results_sensitivity[var_names[[i]]])
  summary(mod_sensitivity[[var_names[i]]]) %>% 
    print()
}

results_sensitivity[["road"]]$value <- results_sensitivity[["road"]]$value / 1000

for (i in 1:length(var_names))
{
  fig <- ggplot(results_sensitivity[[var_names[i]]]) +
    geom_point(aes(x = value, y = mean_prediction), size = 0.3) +
    geom_line(aes(x = value, y = mean_prediction)) +
    lims(y = c(0, 1)) +
    labs(x = var_desciptions[[i]],
         y = "Predicted loss") +
    theme(axis.text = element_text(size = 20),
          panel.background = element_blank(),
          panel.border = element_rect(color = 'black', fill = NA, linewidth = 0.7),
          axis.title = element_text(size = 30),
          panel.grid.major = element_line(color = 'black', linewidth = 0.1, linetype = 2))
  ggsave(paste0("./Figures/sensitivity_analysis/sensitivity_analysis_", var_names[i], ".tif"), fig, device = "tiff", width = 6, height = 6, dpi = 300)
  print(paste0("Sensitivity analysis plotting for ", var_names[i], " is saved."))
}

# Variable importance from Random Forests.
varimp <- rf$variable.importance %>% 
  sort(decreasing = T) %>%
  data.frame() %>% 
  rownames_to_column(var = 'Variables') %>%
  rename(variable = Variables,
         importance = ".") %>%
  mutate(importance = importance / sum(importance))

varimp_plot <- ggplot(data = varimp) +
  geom_segment(aes(x = 0, xend = importance, y = reorder(variable, importance), yend = reorder(variable, importance)), linewidth = 3, color = "#aeb6bf", alpha = 0.3) +
  geom_point(aes(x = importance, y = reorder(variable, importance)), size = 3, color = "#003D7C") +
  scale_y_discrete(labels = c(
    "Typhoon",
    "Rubber",
    "Oil palm",
    "Coconut",
    "Sugarcane",
    "GDP",
    "Regulatory quality",
    "PA",
    "Sea level",
    "Coffee",
    "Rice",
    "Population",
    "Coastline",
    "HF",
    "City",
    "River",
    "Road",
    "Previous loss"
  )) +
  xlab("Relative importance") +
  ylab("Variables") +
  xlim(0.00, 0.60) +
  theme(axis.line.x = element_blank(),
        axis.text = element_text(size = 14),
        panel.background = element_blank(),
        panel.border = element_rect(color = 'black', fill = NA, linewidth = 0.5),
        axis.title = element_text(size = 20),
        axis.ticks.y = element_blank(),
        panel.grid.major.x = element_line(color = 'black', linewidth = 0.1, linetype = 2))

# The accuracy of the predictions.
rf_pred <- predict(rf, data = test_mangrove)
test_mangrove$rf_pred <- rf_pred$predictions

lm_tst <- lm(data = test_mangrove, formula = rf_pred ~ loss)
summary(lm_tst)
RMSE(test_mangrove$rf_pred, test_mangrove$loss) / (max(test_mangrove$loss) - min(test_mangrove$loss))
pred_accuracy_plot <- ggplot(data = test_mangrove) +
  geom_point(aes(x = loss, y = rf_pred), size = 1, colour = "#003D7C", alpha = 0.05, stroke = 0) +
  geom_smooth(aes(x = loss, y = rf_pred), method = "lm", se = T, color = "#EF7C00", linewidth = 0.2, linetype = 'dashed', formula = y ~ x - 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "ivory4") +
  xlab("Actual loss") +
  ylab("Predicted loss") +
  theme(axis.line = element_blank(),
        axis.text = element_text(size = 14),
        panel.background = element_blank(),
        panel.border = element_rect(color = 'black', fill = NA, linewidth = 0.5),
        axis.title = element_text(size = 20),
        panel.grid.major = element_line(color = 'black', linewidth = 0.1, linetype = 2)) +
  annotate("text", x = 0.8, y = 0.15, label= "Slope: 0.93", size = 5) +
  annotate("text", x = 0.8, y = 0.10, label= "R-square: 0.96", size = 5) +
  annotate("text", x = 0.8, y = 0.05, label= "RMSE: 0.07", size = 5)

# Combine the plots.
fig_1 <- ggarrange(varimp_plot, pred_accuracy_plot, ncol = 2, nrow = 1, labels = c("A", "B"))
ggsave("./Figures/fig_1.tif", fig_1, device = "tiff", width = 14, height = 6, dpi = 600)

# Quasi-experiment: propensity score matching. ----------------------------
# Convert the PAs.
pa_cat = function(index)
{
  if (index == 1){iucn_cat = 'Ia'}
  else if (index == 2){iucn_cat = 'Ib'}
  else if (index == 3){iucn_cat = 'II'}
  else if (index == 4){iucn_cat = 'III'}
  else if (index == 5){iucn_cat = 'IV'}
  else if (index == 6){iucn_cat = 'V'}
  else if (index == 7){iucn_cat = 'VI'}
  else if (index == 8){iucn_cat = 'Others'}
  else {iucn_cat = 'non-PA'}
  
  return (iucn_cat)
}
mangrove$PA_cat = apply(matrix(mangrove$PA), MARGIN = 1, FUN = pa_cat)

pa_check = function(iucn_cat)
{
  if (iucn_cat == 'non-PA'){PA = 0}
  else {PA = 1}
  
  return (PA)
}
mangrove$PA = apply(matrix(mangrove$PA_cat), MARGIN = 1, FUN = pa_check)

# Quasi-experimental matching.
mod_f <- as.formula(loss ~ gdp  + hf  + pop  + city + road + PA + coconut + coffee + olp + rice + rubber + sugarcane + river + coastline + sealevel + typhoon + loss_prev)
# According to SCM, with "canonical" adjustment set.
match_f <- as.formula(PA ~ gdp  + pop  + hf  + rq  + city + road + river + coastline + sealevel + typhoon + coconut + coffee + olp + rice + rubber + sugarcane)
match_mod <- matchit(data = mangrove, formula = match_f, method = 'nearest', distance = 'glm', link = 'logit', verbose = T)
mangrove_match <- match.data(match_mod)
write.csv(mangrove_match, "./Data/Mangrove_match_fullobs.csv", row.names = F)

# Calculate the distance matrix by chunk.---------------------------------
# Calculate the distance matrix
loc_mat <- mangrove_standardized %>% 
  dplyr::select(lon, lat) %>% 
  as.matrix()

submat_list <- list()
for (i in 0:floor(nrow(loc_mat)/10000))
{
  submat_list[[i + 1]] <- loc_mat[(i*10000+1):min((i+1)*10000, nrow(loc_mat)), c('lon', 'lat')]
  print(i + 1)
}

for (i in 1:length(submat_list))
{
  col_list <- list()
  for (j in 1:length(submat_list))
  {
    col_list[[j]] <- rdist.earth(submat_list[[i]], submat_list[[j]], miles = F) %>% 
      apply(1, FUN = function(x) mean(x, na.rm = T) * nrow(submat_list[[j]]) / nrow(mangrove_standardized))
    print(paste0(i, "-", j))
  }
  col_mat <- do.call(cbind, col_list) %>% 
    apply(1, FUN = sum)
  print(paste0("submat ", i, " combined."))
  write.csv(col_mat, paste0("./Data/dist_mat/dist_mat_", i, ".csv"), row.names = F)
  print(paste0("submat ", i, " written."))
}

file_list <- list()
for (i in 1:length(submat_list))
{
  file_list[[i]] <- read.csv(paste0("./Data/dist_mat/dist_mat_", i, ".csv"), header = F)
  print(paste0("submat ", i, " read."))
}
file_df <- do.call(rbind, file_list) %>% 
  rename(dist = V1) %>% 
  filter(dist != "x")
mangrove_match$dist <- file_df$dist
write.csv(mangrove_match, "./Data/Mangrove_match_processed.csv", row.names = F)

# Import the processed mangrove_match dataset and fit models. -------------
# Function to convert PA_cat to a continuous variable.
PA_Cat_to_Cont = function(iucn_cat)
{
  if (iucn_cat == 'Ia'){index = 8}
  else if (iucn_cat == 'Ib'){index = 7}
  else if (iucn_cat == 'II'){index = 6}
  else if (iucn_cat == 'III'){index = 5}
  else if (iucn_cat == 'IV'){index = 4}
  else if (iucn_cat == 'V'){index = 3}
  else if (iucn_cat == 'VI'){index = 2}
  else if (iucn_cat == 'Others'){index = 1}
  else {index = 0}
  
  return (index)
}
mangrove_match <- read.csv("./Data/Mangrove_match_processed.csv") %>% 
  mutate(
    PA_cont = apply(matrix(PA_cat), MARGIN = 1, FUN = PA_Cat_to_Cont),
    PA = as.factor(PA),
    PA_cat = as.factor(PA_cat),
    pseudo_group = factor('group_a'),
    ISO3 = as.factor(ISO3)
  ) %>% 
  mutate(PA_cat = relevel(PA_cat, ref = 'non-PA'))
str(mangrove_match)

# Standardize the dataset.
mangrove_standardized <- mangrove_match %>% 
  mutate(
    gdp = scale(gdp)[,1],
    pop = scale(pop)[,1],
    hf = scale(hf)[,1],
    rq = scale(rq)[,1],
    city = scale(city)[,1],
    road = scale(road)[,1],
    river = scale(river)[,1],
    coastline = scale(coastline)[,1],
    sealevel = scale(sealevel)[,1],
    typhoon = scale(typhoon)[,1],
    coconut = scale(coconut)[,1],
    coffee = scale(coffee)[,1],
    olp = scale(olp)[,1],
    rice = scale(rice)[,1],
    rubber = scale(rubber)[,1],
    sugarcane = scale(sugarcane)[,1],
    dist = scale(dist)[,1]
  )
summary(mangrove_standardized)

# Identified the duplicated points.
duplicated_idx <- mangrove_standardized %>% 
  dplyr::select(lon, lat) %>%
  duplicated() %>% 
  which()

# Jitter the duplicated points. 
mangrove_standardized <- mangrove_standardized %>%
  mutate(
    lon = ifelse(row_number() %in% duplicated_idx, jitter(lon), lon),
    lat = ifelse(row_number() %in% duplicated_idx, jitter(lat), lat)
  )

# Import the final dataset for modeling. ----------------------------------
# mangrove_standardized <- read.csv("./Data/Mangrove_match_processed.csv") %>%
#   mutate(
#     PA = as.factor(PA),
#     PA_cat = as.factor(PA_cat),
#     pseudo_group = factor('group_a'),
#     ISO3 = as.factor(ISO3)
#   )

# Fit GLMMs.
glmm1 <- glmmPQL(data = mangrove_standardized, 
                 family = 'binomial',
                 loss ~ PA_cont*rq + gdp + I(gdp^2) + pop + hf + 
                   city + road + river + coastline + 
                   coconut + coffee + olp + rice + rubber + sugarcane +
                   sealevel + typhoon, 
                 random = ~1|ISO3)
summary(glmm1)
saveRDS(glmm1, "./models/glmm1.rds")

glmm1_1 <- glmmPQL(data = mangrove_standardized, 
                 family = 'binomial',
                 loss ~ PA_cat*rq + gdp + I(gdp^2) + pop + hf + 
                   city + road + river + coastline + 
                   coconut + coffee + olp + rice + rubber + sugarcane +
                   sealevel + typhoon, 
                 random = ~ dist|ISO3)
summary(glmm1_1)
saveRDS(glmm1_1, "./models/glmm1_1.rds") # This one should be the correct one!!!

glmm_coef <- summary(glmm1_1)$coefficients$fixed %>% 
  as.data.frame() %>%
  rownames_to_column(var = "var") %>% 
  rename(coef = ".")
write.csv(glmm_coef, "./Data/glmm_coef.csv", row.names = F)

glmm1_1 <- readRDS("./models/glmm1_1.rds")
summary(glmm1_1)

pa_colors <- c("#4b74b2", "#76a2cf", "#a5cbe5", "#dbebf1", "#f2edc3", "#ffdd8b", "#fda368", "#f06a46", "#db3124")

glmm_coef <- read.csv("./Data/glmm_coef.csv") %>% 
  filter(var != "(Intercept)")

# Drivers of mangrove losses
glmm_coef_drivers <- glmm_coef[-grep("PA", glmm_coef$var),] %>% 
  filter(var != "rq") %>%
  mutate(var = factor(var, levels = c(
    'typhoon',
    'sealevel',
    'sugarcane',
    'rubber',
    'rice',
    'olp',
    'coffee',
    'coconut',
    'coastline',
    'river',
    'road',
    'city',
    'hf',
    'pop',
    'I(gdp^2)',
    'gdp'
  ))) %>% 
  mutate(group = factor(group, levels = c(
    'Natural disasters', 
    'Crop suitabilities',
    'Accessibilities',
    'Human activities'
  )))

plot_coef_drivers <- ggplot(glmm_coef_drivers) +
  # geom_point(aes(x = coef, y = var, col = group), size = 1) +
  geom_errorbarh(aes(xmin = coef - sd, xmax = coef + sd, y = var, col = group), linetype = 1, linewidth = 0.5, height = 1) +
  scale_color_manual(values = c("#292643", "#bbaab8", "#44426e", "#e99e75")) +
  theme(
    axis.ticks = element_line(color = 'black', linewidth = 0.1),
    axis.text = element_text(size = 4),
    panel.background = element_blank(),
    panel.border = element_rect(color = 'black', fill = NA, linewidth = 0.6),
    axis.title = element_text(size = 6),
    panel.grid.major = element_line(color = 'grey', linewidth = 0.2, linetype = 2),
    legend.position = "none"
  ) +
  labs(
    x = "Effect size",
    y = "Variable"
  ) +
  scale_y_discrete(labels = c(
    "Storm frequency",
    "Sea level",
    "Sugarcane",
    "Rubber",
    "Rice",
    "Oil palm",
    "Coffee",
    "Coconut",
    "Coastline",
    "River",
    "Road",
    "City",
    "Human footprint",
    "Population",
    "GDP^2",
    "GDP"
  ))
ggsave("./Figures/plot_coef_drivers.tif", plot_coef_drivers, device = "tiff", width = 4.55, height = 4, units = "cm", dpi = 300)

glmm_coef_pa <- glmm_coef[grep("PA", glmm_coef$var),] %>% 
  mutate(var = factor(var, levels = c(
    'PA_catOthers:rq',
    'PA_catOthers',
    'PA_catVI:rq',
    'PA_catVI',
    'PA_catV:rq',
    'PA_catV',
    'PA_catIV:rq',
    'PA_catIV',
    'PA_catIII:rq',
    'PA_catIII',
    'PA_catII:rq',
    'PA_catII',
    'PA_catIb:rq',
    'PA_catIb',
    'PA_catIa:rq',
    'PA_catIa'
  )))

plot_coef_pa <- ggplot(glmm_coef_pa) +
  # geom_point(aes(x = coef, y = var, col = var), size = 1) +
  geom_errorbarh(aes(xmin = coef - sd, xmax = coef + sd, y = var, col = var), linetype = 1, linewidth = 0.5, height = 1) +
  scale_color_manual(values = rep(rev(pa_colors), each = 2)) +
  theme(
    axis.ticks = element_line(color = 'black', linewidth = 0.1),
    axis.text = element_text(size = 4),
    panel.background = element_blank(),
    panel.border = element_rect(color = 'black', fill = NA, linewidth = 0.6),
    axis.title = element_text(size = 6),
    panel.grid.major = element_line(color = 'grey', linewidth = 0.2, linetype = 2),
    legend.position = 'none'
  ) +
  labs(
    x = "Effect size",
    y = "Variable"
  ) +
  scale_y_discrete(labels = c(
    "Others: RQ",
    "Others",
    "VI: RQ",
    "VI",
    "V: RQ",
    "V",
    "IV: RQ",
    "IV",
    "III: RQ",
    "III",
    "II: RQ",
    "II",
    "Ib: RQ",
    "Ib",
    "Ia: RQ",
    "Ia"
  ))
ggsave("./Figures/plot_coef_pa.tif", plot_coef_pa, device = "tiff", width = 4.55, height = 4, units = "cm", dpi = 300)

vis_glmm1_1 <- ggpredict(glmm1_1, terms = c("rq", "PA_cat"), ci_level = 0.1) %>% 
  as.data.frame() %>% 
  mutate(group = factor(group, levels = c("Ia", "Ib", "II", "III", "IV", "V", "VI", "Others", "non-PA")))
write.csv(vis_glmm1_1, "./Data/vis_glmm1_1.csv", row.names = F)


vis_glmm1_1 <- read.csv("./Data/vis_glmm1_1.csv") %>% 
  mutate(group = factor(group, levels = c("Ia", "Ib", "II", "III", "IV", "V", "VI", "Others", "non-PA")))
interact_pa_rq <- ggplot(vis_glmm1_1) +
  geom_point(aes(x = x, y = predicted, color = group), size = 0.3) +
  geom_line(aes(x = x, y = predicted, color = group), linewidth = 0.2) +
  # geom_ribbon(aes(x = x, ymin = conf.low, ymax = conf.high, fill = group), alpha = 0.1) +
  scale_color_manual(values = colors <- pa_colors) +
  labs(x = "Regulatory quality",
       y = "Mangrove loss") +
  theme(axis.line = element_blank(),
        axis.ticks = element_line(color = 'black', linewidth = 0.1),
        axis.text = element_text(size = 6),
        panel.background = element_blank(),
        panel.border = element_rect(color = 'black', fill = NA, linewidth = 0.6),
        axis.title = element_text(size = 6),
        panel.grid.major = element_line(color = 'grey', linewidth = 0.2, linetype = 2),
        legend.key.size = unit(0.1, "cm"),
        legend.title = element_blank(),
        legend.text = element_text(size = 5),
        legend.position = c(0.65, 0.73),
        legend.background = element_blank()
        )
ggsave("./Figures/interact_pa_rq.tif", interact_pa_rq, device = "tiff", width = 4.55, height = 4, units = "cm", dpi = 300)

# GLMM with spaMM package.
# mangrove_standardized$obs <- 1:nrow(mangrove_standardized) %>% as.factor()
# subsamp <- mangrove_standardized %>% 
#   sample_n(10000)
# loss_mat <- cbind(round(1000*subsamp$loss), 1000-round(1000*subsamp$loss))
# mesh <- INLA::inla.mesh.2d(loc = subsamp[,c('x', 'y')], cutoff = 10000)
# glmm2 <- spaMM::fitme(data = subsamp, 
#                       family = binomial(),
#                       # method = "PQL/L",
#                       loss_mat ~ PA_cont*rq + gdp + I(gdp^2) + pop + hf + 
#                         city + road + river + coastline + 
#                         coconut + coffee + olp + rice + rubber + sugarcane +
#                         sealevel + typhoon +
#                         (1|ISO3))
#                         # MaternIMRFa(1 | x + y, mesh = mesh))
# summary(glmm2)
# saveRDS(glmm2, "./models/glmm2.rds")
# plot(simulateResiduals(glmm2))

# A function to test spatial autocorrelation.
SAC_Test <- function(mod, df, corstr)
{
  residuals <- resid(mod) %>% as.vector()
  resid <- data.frame(
    res = residuals,
    res_standardized = residuals / sd(residuals),
    x = df$x,
    y = df$y
  )
  sp::coordinates(resid) <- ~ x + y
  v <- gstat::variogram(res_standardized ~ x + y, data = resid, width = 50000)
  return(v)
}
v_test <- SAC_Test(glmm1_1, mangrove_standardized, "Sph")
test1 <- gstat::fit.variogram(test, model = gstat::vgm("Lin"))
test1 <- gstat::fit.variogram(test, model = gstat::vgm("Sph"))
test_line <- variogramLine(test1, maxdist = max(test$dist))
v_test_plot <- ggplot() +
  geom_point(data = v_test, aes(x = dist, y = gamma), alpha = 0.2, size = 0.5) +
  theme(axis.line = element_blank(),
        axis.text = element_text(size = 6),
        panel.background = element_blank(),
        panel.border = element_rect(color = 'black', fill = NA, linewidth = 0.1),
        axis.title = element_text(size = 8),
        panel.grid.major = element_line(color = 'grey', linewidth = 0.1, linetype = 2))
ggsave("./Figures/v_test.tif", v_test_plot, device = "tiff", width = 14, height = 9, dpi = 600, units = "cm")

resid(glmm1_1)

# Check the spatial autocorrelation, using gstat package (work really well).
glmm1 <- readRDS("./models/glmm1.rds")
resid_glmm1 <- data.frame(
  res = resid(glmm1),
  res_standardized = resid(glmm1) / sd(resid(glmm1)),
  x = mangrove_standardized$x,
  y = mangrove_standardized$y
)
sp::coordinates(resid_glmm1) <- ~ x + y


vario_standardized_glmm1 <- gstat::variogram(res_standardized ~ x + y, data = resid_glmm1, width = 10000, cutoff = 1000000)
saveRDS(vario_standardized_glmm1, "./models/vario_standardized_glmm1.rds")
vario_standardized_glmm1 <- readRDS("./models/vario_standardized_glmm1.rds")
vmod_glmm1_Exp <- gstat::fit.variogram(vario_standardized_glmm1, model = gstat::vgm(psill = 0.6, "Exp", range = 200000, nugget = 0.1))
vmod_glmm1_Gaus <- gstat::fit.variogram(vario_standardized_glmm1, model = gstat::vgm(psill = 0.6, "Gau", range = 200000, nugget = 0.1))
vmod_glmm1_Sph <- gstat::fit.variogram(vario_standardized_glmm1, model = gstat::vgm(psill = 0.6, "Sph", range = 200000, nugget = 0.1))

sac_vmod <- data.frame(
  model = c("Exp", "Gaus", "Sph"),
  vmod = c(attr(vmod_glmm1_Exp, "SSErr"), attr(vmod_glmm1_Gaus, "SSErr"), attr(vmod_glmm1_Sph, "SSErr"))
) %>% 
  arrange(vmod)

preds_standardized_glmm1 <- variogramLine(variomodel_standardized_glmm1, maxdist = max(vario_standardized_glmm1$dist))
ggplot() +
  geom_point(data = vario_standardized_glmm1, aes(x = dist, y = gamma), alpha = 0.2, size = 0.5) +
  geom_line(data = preds_standardized_glmm1, aes(x = dist, y = gamma))

# Fit GLMM with spatial correlation structure.
glmm1_1 <- fitme(data = mangrove_standardized, 
                 family = 'binomial',
                 loss ~ PA_cont*rq + gdp + I(gdp^2) + pop + hf + 
                   city + road + river + coastline + 
                   + coconut + coffee + olp + rice + rubber + sugarcane +
                   sealevel + typhoon + 
                   (1|ISO3) +
                   Matern(1|x + y),
                 HLmethod = "PQL/L")

glmm1_1 <- glmmPQL(data = mangrove_standardized, 
                   family = 'binomial',
                   loss ~ PA_cont*rq + gdp + I(gdp^2) + pop + hf + 
                     city + road + river + coastline + 
                     + coconut + coffee + olp + rice + rubber + sugarcane +
                     sealevel + typhoon,
                   random = ~1|ISO3,
                   correlation = corSpher(form = ~ lon + lat | ISO3))
summary(glmm1_1)
saveRDS(glmm1_1, "./models/glmm1_1.rds")








