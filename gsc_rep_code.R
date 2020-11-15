## Replication code for GSC Module
## Copyright Adam Hearn 2020

## Read in packages
library("pkgbuild")
library("devtools")
library("gsynth")
library("tidyverse")
library("panelView")
library("ggthemes")

## Cleaning data for Pell regression, removing NAs
dta <- read.csv("panel.csv")
dta <- filter(dta, dta >= 0)
dta <- filter(dta, test_optional >= 0)
pell_dta <- filter(dta, pell_perc_ft >= 0)
pell_dta <- filter(pell_dta, adm_rate >= 0)
pell_dta <- filter(pell_dta, ln_avg_grant >= 0)
pell_dta <- filter(pell_dta, ln_tuition_in >= 0)
pell_dta <- filter(pell_dta, ln_ug_fte >= 0)
pell_dta$adm_rate = pell_dta$adm_rate * 100

## Pell Model
pell <- gsynth(pell_perc_ft ~ factor(test_optional) + ln_ug_fte  + adm_rate + ln_avg_grant + ln_tuition_in, 
               data = pell_dta, index = c("unitid","year"), 
               force = "two-way", CV = FALSE, r = c(0,4), se = TRUE, 
               inference = "parametric", nboots = 2500, parallel = TRUE, 
               min.T0 = 6, seed = 123)

## Cleaning data for Minority regression
dta <- read.csv("panel.csv")
min_dta <- filter(dta, perc_min_ft >= 0)
min_dta <- filter(min_dta, adm_rate >= 0)
min_dta <- filter(min_dta, ln_avg_grant >= 0)
min_dta <- filter(min_dta, ln_tuition_in >= 0)
min_dta <- filter(min_dta, ln_ug_fte >= 0)
min_dta$adm_rate = min_dta$adm_rate * 100

min <- gsynth(perc_min_ft ~ factor(test_optional) + ln_ug_fte  + adm_rate + ln_avg_grant + ln_tuition_in, 
               data = min_dta, index = c("unitid","year"), 
               force = "two-way", CV = FALSE, r = c(0,9), se = TRUE, 
               inference = "parametric", nboots = 2500, parallel = TRUE, 
               min.T0 = 11, seed = 123)
## TCA
#### preprocessing
pell_ct = pell$Y.ct.cnt
pell_tr = pell$Y.tr.cnt
min_ct = min$Y.ct.cnt
min_tr = min$Y.tr.cnt
min = as.data.frame(cbind(min_tr, min_ct))
min <- cbind(time = rownames(min), min)
rownames(min) <- 1:nrow(min)
pell = as.data.frame(cbind(pell_tr, pell_ct))
pell <- cbind(time = rownames(pell), pell)
rownames(pell) <- 1:nrow(pell)
ct = merge(min, pell, by = "time", all = TRUE)
ct <- ct %>% 
  mutate(time = as.numeric(time))
ct <- ct[order(ct$time),]
melted = reshape2::melt(ct, id.var = 'time')
melted$group2 <- ifelse(grepl("tr", melted$variable), "Treated avg.", "Est. counterfactual avg.")
melted$variable <- ifelse(grepl("min", melted$variable), "Pct. minority", "Pct. Pell")
melted$variable <- as_factor(melted$variable)
melted$group2 <- as_factor(melted$group2)
melted$variable <- factor(melted$variable, levels=rev(levels(melted$variable)))

#### graphing
plt <- ggplot(data = melted, aes(x = time, y = value)) + 
  geom_rect(xmin = 0, xmax = 5, ymin = 15, ymax = 55, fill = "grey20", alpha = .002) +
  ylim(20, 50) + 
  geom_line(aes(color = variable, linetype = group2), size = 1.125) +
  scale_color_manual(values = c(rgb(0,52,98, max = 255), rgb(0, 117,226, max = 255)), guide_legend(title = "")) +
  #scale_linetype_manual(values = c("longdash", "solid")) +
  geom_vline(xintercept = 0, linetype = "dotted") +
  scale_linetype(guide_legend(order = 20, title = "")) +
  labs(title = "Treated and Counterfactual Averages",
       x = "Time relative to treatment",
       y = "Pct. first-time enrollment") +
  theme(plot.title = element_text(colour = rgb(0,117, 226, max = 255), size =17.5))

## ATT (Pell)
plot(pell, type = "gap", xlim = c(-10, 4)) + 
  theme_bw() + 
  # labs(caption = "ATT refers to average treatment effect on the treated.") +
  scale_x_continuous(breaks = c(-5, -4, -3, -2, -1, 0, 1, 2,3, 4)) +
  geom_vline(xintercept = 0, size = 2, alpha = .2) +
  geom_hline(yintercept = 0, size = 2, alpha = .2) +
  labs(title = "ATT: Pct. full-time first-time students awarded Pell",
       x = "Time relative to treatment",
       y = "Coefficient") +
  theme(plot.title = element_text(colour = rgb(0,0, 0, max = 255), size =17.5)) +
  ylim(-6, 10) + scale_y_continuous(breaks = c(-6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16))

## ATT (Minority)
plot(min, type = "gap") +
  theme_bw() + 
  #labs(caption = "ATT refers to average treatment effect on the treated.") +
  scale_x_continuous(breaks = c(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2,3, 4)) +
  geom_vline(xintercept = 0, size = 1.5, alpha = .2) +
  geom_hline(yintercept = 0, size = 1.5, alpha = .2) +
  labs(title = "Pct. degree-seeking first-time students minority",
       x = "Time relative to treatment",
       y = "Coefficient") +
  theme(plot.title = element_text(colour = rgb(0, 0, 0, max = 255), size =17.5)) +
  ylim(-6, 16) +   scale_y_continuous(breaks = c(-6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16)) 