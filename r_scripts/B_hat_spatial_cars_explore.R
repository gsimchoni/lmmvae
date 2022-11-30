library(tidyverse)
library(scales)
library(patchwork)

B_hat <- read_csv("data/data_for_viz/B_hat_spatial_cars.csv")
coords <- read_csv("data/data_for_viz/new_coords_cars.csv")

# from original cars df
# range(cars_df$long)
# [1] -124.28400  -67.84049
# range(cars_df$lat)
# [1] 24.5552 48.9636

# from states
# range(state$long)
# [1] -124.68134  -67.00742
# range(state$lat)
# [1] 25.12993 49.38323

state <- map_data("state")

df <- tibble(long = coords$lat, lat = coords$long) #mixup between lon/lat locations

df$long <- rescale(df$long, to = range(state$long))
df$lat <- rescale(df$lat, to = range(state$lat))
n_bins <- 20

df$price <-  B_hat$X0
df$odometer <-  B_hat$X2

df <- df %>%
  mutate(long_cut = cut(long, breaks=seq(min(.$long), max(.$long), length.out = n_bins)),
         long_cut = str_sub(long_cut, 2, -2)) %>%
  separate(long_cut, c("long2", "long3"), sep = ",") %>%
  mutate_at(c("long2", "long3"), as.double)

df$long <- (df$long2 + df$long3)/2

df <- df %>%
  mutate(lat_cut = cut(lat, breaks=seq(min(.$lat), max(.$lat), length.out = n_bins)),
         lat_cut = str_sub(lat_cut, 2, -2)) %>%
  separate(lat_cut, c("lat2", "lat3"), sep = ",") %>%
  mutate_at(c("lat2", "lat3"), as.double)

df$lat <- (df$lat2 + df$lat3)/2

p1 <- ggplot(df, aes(long, lat)) +
  geom_raster(aes(fill = price), interpolate=TRUE) +
  labs(x = "Longitude", y = "Latitude") +
  scale_fill_continuous(low = "white", high = "blue") +
  theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(),
        axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank()) + 
  theme_bw() +
  geom_polygon(aes(group = group), data = state, color = "white", alpha = 0.2) +
  guides(fill="none") +
  coord_fixed(1.3)

p2 <- ggplot(df, aes(long, lat)) +
  geom_raster(aes(fill = odometer), interpolate=TRUE) +
  labs(x = "Longitude", y = "") +
  scale_fill_continuous(low = "white", high = "blue") +
  theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(),
        axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank()) + 
  theme_bw() +
  geom_polygon(aes(group = group), data = state, color = "white", alpha = 0.2) +
  guides(fill="none") +
  coord_fixed(1.3)

p <- p1 | p2

ggsave("images/B_hat_spatial_viz.png", p, device = "png", width = 16, height = 8, dpi = 300)
