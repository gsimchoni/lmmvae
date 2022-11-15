library(tidyverse)
library(patchwork)
library(extrafont)

# remotes::install_version("Rttf2pt1", version = "1.3.8")
font_import()
loadfonts(device="win") 


# Multiple categorical simulation viz ---------------------------------------


B <- read_csv("data/data_for_viz/B_pred_categorical.csv")
U <- read_csv("data/data_for_viz/U_pred_categorical.csv")
X <- read_csv("data/data_for_viz/X_reconstructed_categorical.csv")

p_scatter_B1 <- B %>%
  filter(k == 0) %>%
  ggplot(aes(b_true, b_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "", y = "Predicted RE") +
  # xlim(-1, 1) +
  # ylim(-1, 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p_scatter_B2 <- B %>%
  filter(k == 1) %>%
  ggplot(aes(b_true, b_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True RE", y = "") +
  # xlim(-10, 10) +
  # ylim(-10, 10) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p_scatter_B3 <- B %>%
  filter(k == 2) %>%
  ggplot(aes(b_true, b_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "", y = "") +
  # xlim(-1, 1) +
  # ylim(-1, 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p_scatter_U1 <- U %>%
  ggplot(aes(U0_true, U0_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "", y = "Predicted LV") +
  # xlim(-1, 1) +
  # ylim(-1, 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p_scatter_U2 <- U %>%
  ggplot(aes(U1_true, U1_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "", y = "") +
  # xlim(-1, 1) +
  # ylim(-1, 1) +
  theme_bw() +
  theme(aspect.ratio = 1,
        text = element_text(family = "Century", size = 16))

p_scatter_X1 <- X %>%
  ggplot(aes(X0, X0_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "", y = "Predicted X") +
  # xlim(-1, 1) +
  # ylim(-1, 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p_scatter_X2 <- X %>%
  ggplot(aes(X1, X1_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True X", y = "", title="True LV") +
  # xlim(-1, 1) +
  # ylim(-1, 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16), plot.title = element_text(size=16, hjust = 0.5, vjust = 8))

p_scatter_X3 <- X %>%
  ggplot(aes(X2, X2_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = " ", y = "") +
  # xlim(-1, 1) +
  # ylim(-1, 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

# working
p1 <- (p_scatter_B1 | p_scatter_B2 | p_scatter_B3)
p2 <- (p_scatter_U1 | p_scatter_U2)
p3 <- (p_scatter_X1 | p_scatter_X2 | p_scatter_X3)
p <- p1 / p2 / p3

# working for 1 LV
design <- "
  123
  #4#
  567
"

p <- p_scatter_B1 + p_scatter_B2 + p_scatter_B3 + p_scatter_U1 + p_scatter_X1 + p_scatter_X2 + p_scatter_X3 + plot_layout(design = design)

ggsave("images/sim_viz.png", p, device = "png", width = 16, height = 8, dpi = 300)
