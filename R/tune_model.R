###############################################################################
###                             TUNE CONTEXTNET MODEL
###
### Runs:         On HPC cluster.
### Status:       Almost complete - mainly tests and documentation remaining.
### Priority:     Medium.
### User facing:  No.
###############################################################################


#' Get Parameter List from Grid Row
#'
#' @param grid_row Grid row of parameter settings to evaluate
#'
#' @return A list
#' @export
#'
#' @examples
get_row_list <- function(grid_row) {
  l <- list("n_filts" = unlist(grid_row$n_filts),
            "kern_sizes" = unlist(grid_row$kern_sizes),
            "lr" = unlist(grid_row$lr),
            "lambda_cnn" = unlist(grid_row$lambda_cnn),
            "lambda_corr" = unlist(grid_row$lambda_corr),
            "lambda_out" = unlist(grid_row$lambda_out),
            "epochs" = unlist(grid_row$epochs),
            "batch_size" = unlist(grid_row$batch_size),
            "patience" = unlist(grid_row$patience))
  if (!"covars" %in% names(grid_row)) {
    l$covars <- NULL
  } else {
    l$covars <- unlist(grid_row$covars)
  }
  return(l)
}


#' Get Document-Level Activation Metrics (tuning)
#'
#' @inheritParams get_doc_acts
#'
#' @return Data frame with metrics.
#' @export
#'
#' @examples
get_doc_metrics <- function(model, params, embeds, dat) {
  # Document-level activation ranges
  acts <- get_doc_acts(model = model, params = params, embeds = embeds,
                       dat = dat)
  if (any(is.nan(acts$activation)) | any(is.nan(acts$wt))) {
    return(data.frame("act_range" =  as.character(NaN),
                      "max_corr" = NaN))
  }

  ranges <- stats::aggregate(activation ~ filter, acts, range)
  ranges <- round(ranges$activation[, 2] - ranges$activation[, 1], 3)
  ranges <- paste(ranges, collapse = "|")

  # Max correlation between document-level activations
  acts <- tidyr::pivot_wider(acts[, c("filter", "activation", "sample_id")],
                             names_from = "filter", id_cols = "sample_id",
                             values_from = "activation")
  acts <- acts[, 2:ncol(acts)]
  max_corr <- max(stats::cor(acts) - diag(nrow = ncol(acts)))
  return(data.frame("act_range" = ranges, "max_corr" = max_corr))
}


#' Tuning Wrapper for get_top_phrases_quick()
#'
#' @param embeds Word embeddings
#' @param model Trained model
#' @param dat Original data set
#' @inheritParams get_top_phrases_quick
#'
#' @return Short summary of top phrases
#'
#' @examples
get_phrase_metrics <- function(model, params, embeds, tokens, vocab, dat) {
  acts <- get_phrase_acts(model = model, params = params, embeds = embeds,
                          dat = dat)
  if (any(is.nan(acts$activation)) | any(is.nan(acts$wt))) {
    return("NaN")
  }
  return(get_top_phrases_quick(phrase_acts = acts,  tokens = tokens,
                               params = params, vocab = vocab))
}


#' Get Performance Metrics During Parameter Tuning
#'
#' @param model Trained Keras model
#' @param embeds Word embeddings
#' @param dat Input text data and covariates and outcome
#' @param metrics Metrics to get from eval_model
#' @param fold Fold to assess
#' @param covars Covariates for this model
#'
#' @return Performance metrics in a data frame row.
#'
#' @examples
get_tune_metrics <- function(model, embeds, dat, metrics, fold, covars = NULL) {

  if (!is.null(covars)) {  ## Covariate handling
    covs <- as.matrix(dat[, covars])
    res <- eval_model(model, list(embeds[dat$fold == fold, , ],
                                  covs[dat$fold == fold, ]),
                      dat$y[dat$fold == fold], metrics)
  } else {
    res <- eval_model(model, embeds[dat$fold == fold, , ],
                      dat$y[dat$fold == fold], metrics)
  }
  return(res)
}


#' Combine job array tuning results.
#'
#' @param folder_name Name of directory where model files are saved.
#' @param folder_path Path to parent directory where folder_name should be
#'        stored.
#' @param grid Parameter tuning grid.
#'
#' @return No return object, but compiled tuning results will be saved if the
#'         job array was successful.
#' @export
#'
#' @examples
combine_tune_res <- function(folder_path, folder_name, grid) {

  ### Stitch together job array results
  files <- list.files(paste0(file.path(folder_path, folder_name)),
                      full.names = TRUE)
  files <- files[grep("tuning_results_", files)]

  x <- readRDS(files[1])
  for (f in files[2:length(files)]) x <- rbind(x, readRDS(f))

  # Check to see if any rows of the original parameter grid are missing from
  # running out of memory/time
  grid$job <- rep(1:ceiling(nrow(grid) /
                              input_embeds$params$compute_params$job_iters),
                  each = 40)[1:nrow(grid)]
  temp <- merge(grid[, c("job", "id", "run", "n_filts", "kern_sizes")],
                x[, c("id", "run", "train_mse")], all.x = TRUE)

  # Save tuning results if all jobs finished
  missing_jobs <- unique(temp$job[is.na(temp$train_mse)])
  if (identical(missing_jobs, integer(0))) {
    saveRDS(x, paste0(file.path(folder_path, folder_name),
                      "/tuning_results.rds"))
  } else {
    stop(paste0("Missing results from jobs: ",
                paste(missing_jobs, collapse = ", "), "."))
  }
}


#' Aggregate tuning results across folds
#'
#' @param tune_res Parameter grid with tuning results included.
#'
#' @return Aggregated version of tune_res, a data frame.
#' @export
#'
#' @examples
agg_tuning_res <- function(tune_res) {

  # Take the average activation range across both filters and runs of the same
  # parameter setting.
  tune_res$act_range_avg <- apply(do.call(rbind,
                                          strsplit(tune_res$act_range, "|",
                                                   fixed = TRUE)),
                                  1, function(a) mean(as.numeric(a)))

  # Average the other quantitative columns across runs of the same setting.
  quant_cols <- grep("train|val|max_corr|avg", names(tune_res), value = TRUE)
  tune_res_agg <- tune_res %>%
    group_by(id) %>%
    summarise(across(all_of(quant_cols), ~ mean(.x, na.rm = TRUE)),
              id = unique(id))

  tune_res_agg <- tune_res_agg[order(tune_res_agg$val_mse), ]
  return(tune_res_agg)

}


#' Tune ConTextNet Model
#'
#' @param dat Original text data set with outcome `y` (assumes full, will auto
#'        discard test set)
#' @param embeds Text embeddings for `dat$text`
#' @param meta_params Meta-parameters for model training
#' @param grid Grid of parameter settings to evaluate
#' @param tokens Tokens for `dat$text`
#' @param vocab Vocab map from tokenizer
#'
#' @return Data frame, which is grid with model performances filled in.
#' @export
#'
#' @examples
tune_model <- function(dat, embeds, meta_params, grid, tokens, vocab) {

  # Only proceed with training data for tuning.
  embeds <- embeds[dat$fold == "train", , ]
  tokens <- tokens[dat$fold == "train", ]
  dat <- dat[dat$fold == "train", ]
  temp <- dat

  if (meta_params$task == "class") {
    metrics <- c("accuracy", "f1", "mse")
  } else metrics <- "mse"

  # Loop through parameter settings in grid.
  for (i in 1:nrow(grid)) {

    print(paste0("Starting grid row ", i))

    # Grab parameter settings for this grid row.
    these_params <- c(meta_params, get_row_list(grid[i, ]))

    # Switch up test and train folds for cross-validation, train model.
    temp$fold <- ifelse(dat$tune_fold == grid$run[i], "test", "train")
    model <- train_model(temp, embeds, these_params, run_quiet = TRUE)

    # Record performance metrics in grid.
    if (meta_params$task == "class") {  # Add certain metrics for classifiers
      cols <- c("train_mse", "train_acc", "train_f1")
    } else cols <- "train_mse"
    grid[i, cols] <- get_tune_metrics(model = model, embeds = embeds,
                                      dat = temp, metrics = metrics,
                                      fold = "train",
                                      covars = these_params$covars)
    cols <- gsub("train", "val", cols)
    grid[i, cols] <- get_tune_metrics(model = model, embeds = embeds,
                                      dat = temp, metrics = metrics,
                                      fold = "test",
                                      covars = these_params$covars)

    # Record spread of the filter activations on the validation samples and
    # max correlation between document-level filter activations.
    grid[i, c("act_range", "max_corr")] <- get_doc_metrics(model = model,
                                   params = these_params,
                                   embeds = embeds[temp$fold == "test", , ],
                                   dat = temp[temp$fold == "test", ])

    # Finally, return very short summary of filter interpretations on the
    # validation set.
    grid$phrases[i] <- get_phrase_metrics(model = model, params = these_params,
                                  embeds = embeds[temp$fold == "test", , ],
                                  tokens = tokens[temp$fold == "test", ],
                                  vocab = vocab,
                                  dat = temp[temp$fold == "test", ])

    # Free up memory (only partially successful in my past experience, thus the
    # batching).
    rm(list = c("model"))
    keras::k_clear_session()
    tf$keras$backend$clear_session()
    gc()
  }

  return(grid)

}
