# ============================================================
# 05_A3_normality.R
# ERST619 Kaupapa Tuhika 3 — Normality + transformations
# Input : ../../data/PortHills2017_stack.tif, core cells (inside_aoi == 1)
# Output: figures/A3_histograms.png
#         figures/A3_qq_NIR.png
#         tables/A3_normality.csv
# Kurtosis is Pearson (normal = 3), same convention as ArcGIS
# ============================================================

library(terra)
library(ggplot2)
library(dplyr)
library(tidyr)

dir.create("figures", showWarnings = FALSE)
dir.create("tables",  showWarnings = FALSE)

# 1) Read ------------------------------------------------------------
r <- rast("../../data/PortHills2017_stack.tif")
band_names <- c("pre_B2","pre_B3","pre_B4","pre_B5","pre_B6","pre_B7",
                "post_B2","post_B3","post_B4","post_B5","post_B6","post_B7",
                "NBR_pre","NBR_post","dNBR","NDVI","NDMI","BSI",
                "severity","dnbr_data","pre_source","grid_row","grid_col",
                "elev","slope","northness","ephemeral_refugia",
                "pre_analysis_valid","inside_aoi","clean_pre_mask","post_neg_refl","dNBR_cs")
if (!identical(names(r), band_names)) stop("Unexpected main-stack bands; see ../../data/README.md")

spec_vars <- c("pre_B2","pre_B3","pre_B4","pre_B5","pre_B6","pre_B7",
               "NDVI","NDMI","BSI","NBR_pre")
terr_vars <- c("elev","slope","northness")
vars      <- c(spec_vars, terr_vars)

df   <- as.data.frame(r[[c(vars, "inside_aoi")]], na.rm = FALSE)
core <- df %>% filter(!is.na(inside_aoi), inside_aoi == 1) %>% select(all_of(vars))

# 2) Helpers ---------------------------------------------------------
skew_fn <- function(x) { m <- mean(x); mean((x - m)^3) / mean((x - m)^2)^1.5 }
kurt_fn <- function(x) { m <- mean(x); mean((x - m)^4) / mean((x - m)^2)^2 }

# Shift to positive; reflect first if left-skewed so log/sqrt pull the long tail in
prep_pos <- function(x) {
  y <- if (skew_fn(x) < 0) max(x) - x else x
  y - min(y) + 0.01 * diff(range(y))
}

# Yeo-Johnson with lambda chosen by maximum likelihood
yj <- function(x, l) {
  out <- numeric(length(x)); p <- x >= 0
  out[p]  <- if (abs(l) > 1e-8)     ((x[p] + 1)^l - 1) / l           else log(x[p] + 1)
  out[!p] <- if (abs(l - 2) > 1e-8) -((-x[!p] + 1)^(2 - l) - 1) / (2 - l) else -log(-x[!p] + 1)
  out
}
yj_fit <- function(x) {
  ll <- function(l) {
    y <- yj(x, l)
    -length(x) / 2 * log(mean((y - mean(y))^2)) + (l - 1) * sum(sign(x) * log(abs(x) + 1))
  }
  # Wide search range: for reflectance (0-0.5) the ML lambda lies well below -3
  # (NIR about -5.5, blue about -18); a [-3, 5] range stopped at the bound.
  l <- optimize(ll, c(-50, 50), maximum = TRUE)$maximum
  list(lambda = l, y = yj(x, l))
}

# 3) Normality table: raw vs log vs sqrt vs Yeo-Johnson vs z-score ---
norm_tab <- lapply(vars, function(v) {
  x  <- core[[v]]
  lx <- log(prep_pos(x)); sx <- sqrt(prep_pos(x))
  f  <- yj_fit(x); zx <- (x - mean(x)) / sd(x)
  data.frame(variable = v,
             group    = ifelse(v %in% spec_vars, "Pre-fire spectral", "Terrain"),
             skew_raw = skew_fn(x),  kurt_raw = kurt_fn(x),
             skew_log = skew_fn(lx), kurt_log = kurt_fn(lx),
             skew_sqrt = skew_fn(sx), kurt_sqrt = kurt_fn(sx),
             skew_yj  = skew_fn(f$y), kurt_yj = kurt_fn(f$y), yj_lambda = f$lambda,
             skew_z   = skew_fn(zx), kurt_z  = kurt_fn(zx),
             bimodality_coef = (skew_fn(x)^2 + 1) / kurt_fn(x))   # > 0.555 suggests bimodal
})
norm_tab <- bind_rows(norm_tab) %>% mutate(across(where(is.numeric), ~ round(.x, 2)))
write.csv(norm_tab, "tables/A3_normality.csv", row.names = FALSE)
print(norm_tab)

# 4) Histograms with fitted normal curve -----------------------------
long <- core %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  mutate(variable = factor(variable, levels = vars))

curves <- long %>%
  group_by(variable) %>%
  summarise(mu = mean(value), s = sd(value), lo = min(value), hi = max(value),
            n = n(), bw = (hi - lo) / 40, .groups = "drop") %>%
  rowwise() %>%
  do(data.frame(variable = .$variable,
                x = seq(.$lo, .$hi, length.out = 200),
                y = dnorm(seq(.$lo, .$hi, length.out = 200), .$mu, .$s) * .$n * .$bw))

labs_df <- norm_tab %>%
  mutate(variable = factor(variable, levels = vars),
         lab = paste0("skew ", sprintf("%.2f", skew_raw), "\nkurt ", sprintf("%.2f", kurt_raw)))

p_hist <- ggplot(long, aes(value)) +
  geom_histogram(aes(fill = variable %in% terr_vars), bins = 40, colour = NA) +
  geom_line(data = curves, aes(x, y), colour = "#B5451F", linewidth = 0.5) +
  geom_text(data = labs_df, aes(x = Inf, y = Inf, label = lab),
            hjust = 1.05, vjust = 1.2, size = 2.6, colour = "#3A2A22") +
  facet_wrap(~ variable, scales = "free", ncol = 5) +
  scale_fill_manual(values = c(`FALSE` = "#E6C9A8", `TRUE` = "#E6A96A"),
                    labels = c("Pre-fire spectral", "Terrain"), name = NULL) +
  labs(title = "Histograms of predictor and ancillary variables (core cells)",
       subtitle = "Red line = normal curve with the same mean and SD; kurtosis: normal = 3",
       x = NULL, y = "Cells") +
  theme_minimal(base_size = 10) +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", colour = "#3A2A22"))
ggsave("figures/A3_histograms.png", p_hist, width = 11, height = 6.5, dpi = 300)

# 5) Q-Q plot for the one clearly skewed band: NIR raw vs Yeo-Johnson
nir <- core$pre_B5
qq  <- bind_rows(
  data.frame(version = "Raw NIR",
             sample = sort((nir - mean(nir)) / sd(nir))),
  data.frame(version = "Yeo-Johnson NIR",
             sample = sort({ y <- yj_fit(nir)$y; (y - mean(y)) / sd(y) }))
) %>% group_by(version) %>%
  mutate(theoretical = qnorm(ppoints(n()))) %>% ungroup()

p_qq <- ggplot(qq, aes(theoretical, sample)) +
  geom_abline(slope = 1, intercept = 0, colour = "#8A756A", linetype = "dashed") +
  geom_point(size = 0.4, alpha = 0.4, colour = "#B5451F") +
  facet_wrap(~ version) +
  labs(title = "Q-Q plots: NIR before and after Yeo-Johnson",
       x = "Theoretical quantiles", y = "Standardised sample quantiles") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold", colour = "#3A2A22"))
ggsave("figures/A3_qq_NIR.png", p_qq, width = 8, height = 4.2, dpi = 300)

cat("Done: figures/A3_histograms.png, figures/A3_qq_NIR.png, tables/A3_normality.csv\n")
