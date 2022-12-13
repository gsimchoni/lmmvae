library(tidyverse)
library(lubridate)
library(patchwork)
library(extrafont)

# remotes::install_version("Rttf2pt1", version = "1.3.8")
font_import()
loadfonts(device="win") 

X_te <- read_csv("data/data_for_viz/X_te_lmmvae_rossmann.csv")
X_pred_lmmvae <- read_csv("data/data_for_viz/X_recon_te_lmmvae_rossmann.csv")
X_pred_vae <- read_csv("data/data_for_viz/X_recon_te_vae_rossmann.csv")

# Sales
s = 6.49116622e-01
m = 1.74512881e+00
n_stores <- 5
stores <- 0:(n_stores - 1)

df_orig <- X_te %>%
  select(orig_id, z0, t, y) %>%
  arrange(z0, t) %>%
  mutate(y = y * s + m) %>%
  filter(z0 %in% stores)

df_orig$t <- rep(c(ym("2015-02"), ym("2015-03"), ym("2015-04"), ym("2015-05"), ym("2015-06"), ym("2015-07")), n_stores)

df_lmmvae <- X_pred_lmmvae %>%
  select(orig_id, z0, t, y) %>%
  arrange(z0, t) %>%
  mutate(y = y * s + m) %>%
  filter(z0 %in% stores) %>%
  select(orig_id, lmmvae = y)

df_vae <- X_pred_vae %>%
  select(orig_id, z0, t, y) %>%
  arrange(z0, t) %>%
  mutate(y = y * s + m) %>%
  filter(z0 %in% stores) %>%
  select(orig_id, vae = y)

df_y <- inner_join(inner_join(df_orig, df_lmmvae, by = "orig_id"), df_vae, by = "orig_id")

df_y_long <- df_y %>%
  pivot_longer(cols = c(y, lmmvae, vae), names_to = "method", values_to = "sales") %>%
  mutate(z0 = str_c("Store ", z0 + 1))

p1 <- df_y_long %>%
  ggplot(aes(t, sales, color = method)) +
  geom_line() +
  facet_grid(z0 ~ .) +
  theme_bw() +
  labs(x = "", y = "Sales ($100K)") +
  scale_color_manual(values = c("red", "green", "blue"), labels = c("LMMVAE", "VAE", "TRUE")) +
  guides(color = "none") +
  scale_x_date(date_labels = "%b") +
  theme(text = element_text(family = "Century", size=16))

# SchoolHoliday
s = 6.06048027e+00
m = 5.59514699e+00
n_stores <- 5
stores <- 0:(n_stores - 1)
# stores <- sample(unique(X_te$z0), n_stores, replace = FALSE)

df_orig <- X_te %>%
  select(orig_id, z0, t, y = schoolholiday) %>%
  arrange(z0, t) %>%
  mutate(y = y * s + m) %>%
  filter(z0 %in% stores)

df_orig$t <- rep(c(ym("2015-02"), ym("2015-03"), ym("2015-04"), ym("2015-05"), ym("2015-06"), ym("2015-07")), n_stores)

df_lmmvae <- X_pred_lmmvae %>%
  select(orig_id, z0, t, y = schoolholiday) %>%
  arrange(z0, t) %>%
  mutate(y = y * s + m) %>%
  filter(z0 %in% stores) %>%
  select(orig_id, lmmvae = y)

df_vae <- X_pred_vae %>%
  select(orig_id, z0, t, y = schoolholiday) %>%
  arrange(z0, t) %>%
  mutate(y = y * s + m) %>%
  filter(z0 %in% stores) %>%
  select(orig_id, vae = y)

df_y <- inner_join(inner_join(df_orig, df_lmmvae, by = "orig_id"), df_vae, by = "orig_id")

df_y_long <- df_y %>%
  pivot_longer(cols = c(y, lmmvae, vae), names_to = "method", values_to = "school_holiday") %>%
  mutate(z0 = str_c("Store ", z0 + 1))

p2 <- df_y_long %>%
  ggplot(aes(t, school_holiday, color = method)) +
  geom_line() +
  facet_grid(z0 ~ .) +
  theme_bw() +
  labs(x = "", y = "No. of School Holiday Days", color = "") +
  scale_color_manual(values = c("red", "green", "blue"), labels = c("LMMVAE", "VAE", "TRUE")) +
  scale_x_date(date_labels = "%b") +
  theme(text = element_text(family = "Century", size=16))


p <- p1 | p2

ggsave("images/X_te_reconstruction_longitudinal.png", p, device = "png", width = 16, height = 8, dpi = 300)
