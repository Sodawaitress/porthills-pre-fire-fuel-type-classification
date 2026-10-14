# ============================================================
# 04_A3_outliers.R
# ERST619 Kaupapa Tuhika 3 — Outliers (IQR box plots + scatterplots)
# Input : ../../data/PortHills2017_stack.tif (32 bands, 30 m NZTM grid)
# Scope : core cells only (inside_aoi == 1, i.e. edge_mix == 0)
# Output: figures/A3_boxplots_IQR.png
#         figures/A3_scatter_red_nir.png
#         figures/A3_scatter_northness_nir.png
#         tables/A3_outlier_summary.csv
#         tables/A3_outlier_types.csv
#         outputs/A3_outlier_type.tif   (for the ArcGIS map)
# ============================================================

library(terra)
library(ggplot2)
library(dplyr)
library(tidyr)

dir.create("figures", showWarnings = FALSE)
dir.create("tables",  showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)

# 1) Read the stack --------------------------------------------------
r <- rast("../../data/PortHills2017_stack.tif")

band_names <- c("pre_B2","pre_B3","pre_B4","pre_B5","pre_B6","pre_B7",
                "post_B2","post_B3","post_B4","post_B5","post_B6","post_B7",
                "NBR_pre","NBR_post","dNBR","NDVI","NDMI","BSI",
                "severity","dnbr_data","pre_source","grid_row","grid_col",
                "elev","slope","northness","ephemeral_refugia",
                "pre_analysis_valid","inside_aoi","clean_pre_mask","post_neg_refl","dNBR_cs")
if (!identical(names(r), band_names)) stop("Unexpected main-stack bands; see ../../data/README.md")

# A3 predictors (pre-fire spectral) + ancillary (terrain)
spec_vars <- c("pre_B2","pre_B3","pre_B4","pre_B5","pre_B6","pre_B7",
               "NDVI","NDMI","BSI","NBR_pre")
terr_vars <- c("elev","slope","northness")
vars      <- c(spec_vars, terr_vars)

df <- as.data.frame(r[[c(vars, "inside_aoi", "pre_source")]],
                    cells = TRUE, na.rm = FALSE)
core <- df %>% filter(!is.na(inside_aoi), inside_aoi == 1)
cat("Core cells:", nrow(core), " (", nrow(core) * 0.09, "ha )\n")

# 2) IQR fences per variable -----------------------------------------
fences <- lapply(vars, function(v) {
  x  <- core[[v]]
  q  <- quantile(x, c(0.25, 0.75), na.rm = TRUE)
  iq <- q[2] - q[1]
  lo <- q[1] - 1.5 * iq;  hi <- q[2] + 1.5 * iq
  data.frame(variable = v,
             Q1 = q[1], Q3 = q[2], IQR = iq,
             lower_fence = lo, upper_fence = hi,
             n_low  = sum(x < lo, na.rm = TRUE),
             n_high = sum(x > hi, na.rm = TRUE),
             n_extreme_3IQR = sum(x < q[1] - 3 * iq | x > q[2] + 3 * iq, na.rm = TRUE),
             pct_outlier = round(100 * mean(x < lo | x > hi, na.rm = TRUE), 2),
             row.names = NULL)
})
fences <- bind_rows(fences)
write.csv(fences, "tables/A3_outlier_summary.csv", row.names = FALSE)
print(fences)

hi_of <- function(v) fences$upper_fence[fences$variable == v]
lo_of <- function(v) fences$lower_fence[fences$variable == v]

# 3) Outlier types (order matters: first match wins) -----------------
core <- core %>%
  mutate(type = case_when(
    pre_B5 < lo_of("pre_B5") ~ "Topographic shadow (low NIR)",
    pre_B5 > hi_of("pre_B5") ~ "Dense canopy (high NIR)",
    pre_B2 > hi_of("pre_B2") | pre_B3 > hi_of("pre_B3") |
      pre_B4 > hi_of("pre_B4") | pre_B6 > hi_of("pre_B6") |
      pre_B7 > hi_of("pre_B7") ~ "Bright sparse surface",
    slope > hi_of("slope")   ~ "Steep terrain",
    TRUE                     ~ "Within fences"
  ))

type_levels <- c("Within fences","Dense canopy (high NIR)","Bright sparse surface",
                 "Topographic shadow (low NIR)","Steep terrain")
core$type <- factor(core$type, levels = type_levels)

type_tab <- core %>%
  count(type) %>%
  mutate(ha = n * 0.09,
         pct = round(100 * n / sum(n), 2),
         jan16_share_pct = sapply(type, function(t)
           round(100 * mean(core$pre_source[core$type == t] == 1), 1)),
         decision = ifelse(type == "Within fences", "-", "Keep + flag"))
write.csv(type_tab, "tables/A3_outlier_types.csv", row.names = FALSE)
print(type_tab)

# 4) Colours (same family as the timeline) ---------------------------
type_cols <- c("Within fences"                = "#CFC4BA",
               "Dense canopy (high NIR)"      = "#2E6B3A",
               "Bright sparse surface"        = "#E7A33E",
               "Topographic shadow (low NIR)" = "#3A4F7A",
               "Steep terrain"                = "#B5451F")

# 5) Box plots (1.5 x IQR whiskers) ----------------------------------
long <- core %>%
  select(all_of(vars)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  mutate(variable = factor(variable, levels = vars),
         group = ifelse(variable %in% spec_vars, "Pre-fire spectral", "Terrain"))

p_box <- ggplot(long, aes(x = "", y = value, fill = group)) +
  geom_boxplot(coef = 1.5, outlier.size = 0.6, outlier.alpha = 0.5,
               outlier.colour = "#B5451F", width = 0.5) +
  facet_wrap(~ variable, scales = "free_y", ncol = 5) +
  scale_fill_manual(values = c("Pre-fire spectral" = "#F0D8BA", "Terrain" = "#E6A96A")) +
  labs(title = "Distribution of predictor and ancillary variables (core cells)",
       subtitle = paste0("Whiskers = 1.5 x IQR; red points = outliers; n = ",
                         format(nrow(core), big.mark = ","), " cells (30 m)"),
       x = NULL, y = NULL, fill = NULL) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "top",
        panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", colour = "#3A2A22"))
ggsave("figures/A3_boxplots_IQR.png", p_box, width = 11, height = 6.5, dpi = 300)

# 6) Scatterplots ----------------------------------------------------
bg <- core %>% filter(type == "Within fences")
fg <- core %>% filter(type != "Within fences")

# 6a) Red vs NIR: spectral feature space
p_rn <- ggplot() +
  geom_point(data = bg, aes(pre_B4, pre_B5), colour = type_cols[1],
             size = 0.4, alpha = 0.4) +
  geom_point(data = fg, aes(pre_B4, pre_B5, colour = type), size = 0.8, alpha = 0.8) +
  geom_hline(yintercept = c(lo_of("pre_B5"), hi_of("pre_B5")),
             linetype = "dashed", colour = "#8A756A") +
  scale_colour_manual(values = type_cols[-1], drop = FALSE) +
  labs(title = "Pre-fire Red vs NIR reflectance",
       subtitle = "Dashed lines = NIR IQR fences",
       x = "Red (SR_B4)", y = "NIR (SR_B5)", colour = "Outlier type") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold", colour = "#3A2A22"))
ggsave("figures/A3_scatter_red_nir.png", p_rn, width = 7.5, height = 5.5, dpi = 300)

# 6b) Northness vs NIR: illumination effect
p_nn <- ggplot() +
  geom_point(data = bg, aes(northness, pre_B5), colour = type_cols[1],
             size = 0.4, alpha = 0.4) +
  geom_point(data = fg, aes(northness, pre_B5, colour = type), size = 0.8, alpha = 0.8) +
  scale_colour_manual(values = type_cols[-1], drop = FALSE) +
  labs(title = "Northness vs NIR reflectance",
       subtitle = "Southern hemisphere: north-facing (+1) slopes are sunlit",
       x = "Northness (cos aspect)", y = "NIR (SR_B5)", colour = "Outlier type") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold", colour = "#3A2A22"))
ggsave("figures/A3_scatter_northness_nir.png", p_nn, width = 7.5, height = 5.5, dpi = 300)

# 7) Outlier-type raster for the ArcGIS map --------------------------
# 0 = within fences, 1 = dense canopy, 2 = bright surface, 3 = shadow, 4 = steep
out_r <- rast(r, nlyrs = 1)
vals  <- rep(NA_integer_, ncell(out_r))
vals[core$cell] <- as.integer(core$type) - 1L
values(out_r) <- vals
names(out_r) <- "outlier_type"
writeRaster(out_r, "outputs/A3_outlier_type.tif", overwrite = TRUE,
            datatype = "INT1U", NAflag = 255)

cat("Done: figures/, tables/, outputs/A3_outlier_type.tif\n")
