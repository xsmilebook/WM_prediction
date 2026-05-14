rm(list = ls())

library(R.matlab)
library(ggplot2)
library(grid)

project_folder <- "/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction"
prediction_folder <- file.path(project_folder, "data", "ABCD", "prediction")
summary_file <- file.path(prediction_folder, "V_holdout_partial_results_total_multi_targets.csv")
out_dir <- file.path(project_folder, "results", "V_holdout")

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

target_configs <- data.frame(
  target_id = c(
    "nihtbx_totalcomp_uncorrected",
    "nihtbx_cryst_uncorrected",
    "nihtbx_fluidcomp_uncorrected",
    "General",
    "Ext",
    "ADHD"
  ),
  target_label = c(
    "Total Cognition",
    "Crystallized Cognition",
    "Fluid Cognition",
    "General Psychopathology",
    "Externalizing Symptoms",
    "ADHD Symptoms"
  ),
  stringsAsFactors = FALSE
)

fc_configs <- data.frame(
  fc_type = c("GGFC", "GWFC", "WWFC"),
  fc_label = c("GG", "GW", "WW"),
  fc_color = c("#2F2F2F", "#2C7FB8", "#D95F0E"),
  corr_col = c("GG_corr", "GW_corr", "WW_corr"),
  p_col = c("GG_empirical_p", "GW_empirical_p", "WW_empirical_p"),
  partial_corr_col = c(NA, "GW_partial_corr", "WW_partial_corr"),
  partial_p_col = c(NA, "GW_partial_empirical_p", "WW_partial_empirical_p"),
  stringsAsFactors = FALSE
)

format_pperm_expr <- function(p_value) {
  if (is.na(p_value)) {
    return("italic(P)*plain('perm')==NA")
  }
  if (p_value < 0.001) {
    return("italic(P)*plain('perm')<0.001")
  }
  sprintf("italic(P)*plain('perm')==%.3f", p_value)
}

pick_mat_field <- function(mat_object, field_names, file_path) {
  valid_name <- field_names[field_names %in% names(mat_object)][1]
  if (is.na(valid_name)) {
    stop("Missing field ", paste(field_names, collapse = "/"), " in ", file_path)
  }
  as.numeric(as.vector(mat_object[[valid_name]]))
}

list_time_dirs <- function(base_dir) {
  time_dirs <- list.dirs(base_dir, recursive = FALSE, full.names = TRUE)
  time_dirs <- time_dirs[basename(time_dirs) %in% list.files(base_dir, pattern = "^Time_[0-9]+$", full.names = FALSE)]
  if (!length(time_dirs)) {
    return(character(0))
  }
  time_ids <- as.integer(sub("^Time_", "", basename(time_dirs)))
  time_dirs[order(time_ids)]
}

load_observed_holdout <- function(target_id, fc_type) {
  base_dir <- file.path(
    prediction_folder,
    target_id,
    "V_holdout",
    "RegressCovariates_Holdout"
  )
  time_dirs <- list_time_dirs(base_dir)
  if (!length(time_dirs)) {
    stop("No observed Time_* folder found in ", base_dir)
  }

  for (time_dir in time_dirs) {
    mat_file <- file.path(time_dir, fc_type, "Holdout_Score.mat")
    if (!file.exists(mat_file)) {
      next
    }

    mat_data <- readMat(mat_file)
    actual_score <- pick_mat_field(mat_data, c("Test_Score", "Test.Score"), mat_file)
    predicted_score <- pick_mat_field(mat_data, c("Predict_Score", "Predict.Score"), mat_file)

    complete_mask <- !is.na(actual_score) & !is.na(predicted_score)
    plot_data <- data.frame(
      Actual = actual_score[complete_mask],
      Predicted = predicted_score[complete_mask]
    )

    if (!nrow(plot_data)) {
      next
    }

    return(list(
      data = plot_data,
      time_label = basename(time_dir)
    ))
  }

  stop("No valid Holdout_Score.mat found for ", target_id, " / ", fc_type)
}

build_annotation_exprs <- function(summary_row, fc_config) {
  mean_line <- sprintf(
    "plain(Mean)~italic(r)==%.3f*' '~%s",
    summary_row[[fc_config$corr_col]],
    format_pperm_expr(summary_row[[fc_config$p_col]])
  )

  if (is.na(fc_config$partial_corr_col)) {
    return(list(mean_line = mean_line, partial_line = NULL))
  }

  partial_line <- sprintf(
    "plain(Partial)~italic(r)==%.2f*' '~%s",
    summary_row[[fc_config$partial_corr_col]],
    format_pperm_expr(summary_row[[fc_config$partial_p_col]])
  )

  list(mean_line = mean_line, partial_line = partial_line)
}

get_single_axis_spec <- function(values) {
  value_range <- range(values, na.rm = TRUE)
  span <- diff(value_range)
  if (!is.finite(span) || span == 0) {
    span <- 1
  }
  breaks <- pretty(value_range, n = 5)
  if (!length(breaks)) {
    breaks <- pretty(c(value_range[1] - 0.5, value_range[2] + 0.5), n = 5)
  }
  break_step <- if (length(breaks) > 1) min(diff(breaks)) else span
  limits <- c(
    value_range[1] - 0.5 * break_step,
    value_range[2] + 0.5 * break_step
  )
  tol <- span * 1e-8
  breaks <- breaks[breaks >= (limits[1] - tol) & breaks <= (limits[2] + tol)]
  if (!length(breaks)) {
    breaks <- pretty(value_range, n = 5)
  }
  list(
    limits = limits,
    breaks = breaks
  )
}

summary_df <- read.csv(summary_file, stringsAsFactors = FALSE)
font_family <- "Arial"

for (target_idx in seq_len(nrow(target_configs))) {
  target_id <- target_configs$target_id[target_idx]
  target_label <- target_configs$target_label[target_idx]

  summary_row <- summary_df[summary_df$targetStr == target_id, , drop = FALSE]
  if (!nrow(summary_row)) {
    stop("Target ", target_id, " not found in ", summary_file)
  }

  for (fc_idx in seq_len(nrow(fc_configs))) {
    fc_config <- fc_configs[fc_idx, ]
    holdout_res <- load_observed_holdout(target_id, fc_config$fc_type)
    plot_data <- holdout_res$data
    x_axis_spec <- get_single_axis_spec(plot_data$Actual)
    y_axis_spec <- get_single_axis_spec(plot_data$Predicted)
    annotation_exprs <- build_annotation_exprs(summary_row, fc_config)

    x_pos <- x_axis_spec$limits[1] + 0.05 * diff(x_axis_spec$limits)
    y_pos_partial <- y_axis_spec$limits[1] + 0.05 * diff(y_axis_spec$limits)
    y_pos_mean <- if (is.null(annotation_exprs$partial_line)) {
      y_pos_partial
    } else {
      y_axis_spec$limits[1] + 0.11 * diff(y_axis_spec$limits)
    }

    g <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
      geom_point(
        color = fc_config$fc_color,
        alpha = 0.32,
        size = 2.1,
        shape = 16
      ) +
      geom_smooth(
        method = "lm",
        formula = y ~ x,
        se = TRUE,
        color = fc_config$fc_color,
        fill = fc_config$fc_color,
        alpha = 0.18,
        linewidth = 1.2
      ) +
      theme_classic(base_size = 13, base_family = font_family) +
      theme(
        plot.background = element_rect(fill = "transparent", colour = NA),
        panel.background = element_rect(fill = "transparent", colour = NA),
        plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
        axis.title = element_text(size = 16),
        axis.text = element_text(size = 13, colour = "black"),
        axis.ticks.length = unit(2.5, "pt"),
        plot.margin = margin(6, 6, 6, 6),
        legend.position = "none"
      ) +
      labs(
        x = paste("Actual", target_label, "Score"),
        y = paste("Predicted", target_label, "Score")
      ) +
      scale_x_continuous(
        breaks = x_axis_spec$breaks
      ) +
      scale_y_continuous(
        breaks = y_axis_spec$breaks
      ) +
      coord_cartesian(
        xlim = x_axis_spec$limits,
        ylim = y_axis_spec$limits,
        expand = FALSE
      ) +
      annotate(
        "text",
        x = x_pos,
        y = y_pos_mean,
        label = annotation_exprs$mean_line,
        parse = TRUE,
        hjust = 0,
        vjust = 0,
        size = 4.8,
        colour = "black",
        family = font_family
      )

    if (!is.null(annotation_exprs$partial_line)) {
      g <- g +
        annotate(
          "text",
          x = x_pos,
          y = y_pos_partial,
          label = annotation_exprs$partial_line,
          parse = TRUE,
          hjust = 0,
          vjust = 0,
          size = 4.8,
          colour = "black",
          family = font_family
        )
    }

    file_stub <- file.path(
      out_dir,
      paste0(target_id, "_", fc_config$fc_label, "_holdout_scatter")
    )

    ggsave(paste0(file_stub, ".tif"), g, width = 6, height = 6, units = "in", dpi = 300, bg = "transparent")
    ggsave(paste0(file_stub, ".svg"), g, width = 6, height = 6, units = "in", bg = "transparent")
    ggsave(
      paste0(file_stub, ".pdf"),
      g,
      device = grDevices::cairo_pdf,
      width = 6,
      height = 6,
      units = "in",
      bg = "transparent"
    )

    message(
      "Saved ", target_id, " / ", fc_config$fc_label,
      " using ", holdout_res$time_label
    )
  }
}
