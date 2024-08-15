#' Tune ConTextNet Model
#'
#' @param inputs Model training inputs from embed()
#'
#' @return Data frame, which is grid with model performances filled in.
#' @export
#'
#' @examples
tune_model <- function(inputs) {
  ### TODO: Incorporate tuning progress into future log system.
  ### TODO: I don't like overwriting fold - change train_model to accept
  ### embeds, dat, params directly instead.

  # Only proceed with training data for tuning.
  inputs$embeds <- inputs$embeds[inputs$dat$fold == "train", , ]
  inputs$dat <- inputs$dat[inputs$dat$fold == "train", ]
  inputs$dat$fold <- inputs$dat$tune_fold

  grid <- inputs$grid

  # Loop through parameter settings in grid
  for (i in 1:nrow(grid)) {
    these_params <- list("n_filts" = unlist(grid$n_filts[i]),
                         "kern_sizes" = unlist(grid$kern_sizes[i]),
                         "lr" = unlist(grid$lr[i]),
                         "lambda_cnn" = unlist(grid$lambda_cnn[i]),
                         "lambda_corr" = unlist(grid$lambda_corr[i]),
                         "lambda_out" = unlist(grid$lambda_out[i]),
                         "epochs" = unlist(grid$epochs[i]),
                         "batch_size" = unlist(grid$batch_size[i]),
                         "patience" = unlist(grid$patience[i]),
                         "covars" = unlist(grid$covars[i]))
    inputs$params <- c(inputs$params, these_params)
    inputs$dat$fold <- ifelse(inputs$dat$tune_fold == grid$run[i],
                              "test", "train")

    model <- 1 ### TODO: Stopped here!

      # model_res <- run(params, dat_train = dat_train, dat_val = dat_val,
      #                  return_model = TRUE, return_acc = TRUE, run_quiet = TRUE,
      #                  print_performance = FALSE, save_path = NULL)
      #
      # # Record model performance metrics
      # this_res[f, c("train_metric", "val_metric")] <- c(model_res$train,
      #                                                   model_res$val)
      #
      # # Also want to evaluate spread of the filter activations on the val samples,
      # # and max correlation between max pooled activations for each filter.
      # doc_act <- get_doc_acts(model = model_res$model, dat = dat_val,
      #                         params = params)
      # ranges <- paste(round(apply(doc_act, 2, function(i) max(i) - min(i)), 5),
      #                 collapse = ", ")
      # this_res[f, "max_pool_metric"] <- ranges
      #
      # corr <- cor(doc_act) - diag(nrow = ncol(doc_act))
      # this_res[f, "max_filter_corr"] <- max(corr)
      #
      # # Record if this run returned NaN for the validation metric
      # this_res[f, "num_nan"] <- as.integer(is.nan(model_res$val))
      #
      # # Free up memory (only partially successful, seems to be some amount of
      # # memory that I can't get rid of...)
      # rm(list = c("dat_train", "dat_val", "model_res", "doc_act"))
      # k_clear_session()
      # tf$keras$backend$clear_session()
      # gc()
    }

    # grid[i, c("train_metric", "val_metric")] <- c(mean(this_res$train_metric,
    #                                                    na.rm = TRUE),
    #                                               mean(this_res$val_metric,
    #                                                    na.rm = TRUE))
    # grid[i, "max_pool_metric"] <- paste(this_res$max_pool_metric, collapse = "|")
    # grid[i, "max_filter_corr"] <- paste(this_res$max_filter_corr, collapse = "|")
    # grid[i, "num_nan"] <- sum(this_res$num_nan, na.rm = TRUE)

    # # Update saved file every intermediate 10 rows between start/end.
    # if (i %% 10 == 0 & ! i %in% c(args$start_row, args$end_row)) {
    #   write.csv(grid, paste0("param_tuning/", args$folder, "/grid.csv"))
    # }

}
