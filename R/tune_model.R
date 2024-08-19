#' Tune ConTextNet Model
#'
#' @param dat Original text data set with outcome `y`
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
  ### TODO: Incorporate tuning progress into future log system.
  ### TODO: Need to shuffle tune_fold, shouldn't be ordered.
  ### TODO: Figure out if memory clearing is possible without adverse user-side
  ### effects?

  # Only proceed with training data for tuning.
  embeds <- embeds[dat$fold == "train", , ]
  tokens <- tokens[dat$fold == "train", ]
  dat <- dat[dat$fold == "train", ]
  temp_dat <- dat

  if (meta_params$task == "class") {
    metrics <- c("accuracy", "f1", "mse")
  } else metrics <- "mse"

  # Loop through parameter settings in grid.
  for (i in 1:nrow(grid)) {

    # Grab parameter settings for this grid row.
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
    these_params <- c(meta_params, these_params)

    # Switch up test and train folds for cross-validation, train model.
    temp_dat$fold <- ifelse(dat$tune_fold == grid$run[i], "test", "train")
    model <- train_model(temp_dat, embeds, these_params, run_quiet = TRUE)

    # Record performance metrics in grid.
    if (meta_params$task == "class") {
      cols <- c("train_mse", "train_acc", "train_f1")
    } else cols <- "train_mse"
    grid[i, cols] <- eval_model(model, embeds[temp_dat$fold == "train", , ],
                                temp_dat$y[temp_dat$fold == "train"], metrics)
    cols <- gsub("train", "val", cols)
    grid[i, cols] <- eval_model(model, embeds[temp_dat$fold == "test", , ],
                                temp_dat$y[temp_dat$fold == "test"], metrics)


    # Evaluate spread of the filter activations on the validation samples,
    # record this in grid as well.
    acts <- get_doc_acts(model = model, params = these_params,
                         embeds = embeds[temp_dat$fold == "test", , ],
                         dat = temp_dat[temp_dat$fold == "test", ])

    ranges <- stats::aggregate(activation ~ filter, acts, range)
    ranges <- round(ranges$activation[, 2] - ranges$activation[, 1], 3)
    grid$act_range[i] <- paste(ranges, collapse = "|")

    # Also record the max correlation between document-level filter activations.
    acts <- tidyr::pivot_wider(acts[, c("filter", "activation", "sample_id")],
                               names_from = "filter", id_cols = "sample_id",
                               values_from = "activation")
    acts <- acts[, 2:ncol(acts)]
    grid$max_corr[i] <-  max(stats::cor(acts) - diag(nrow = ncol(acts)))

    # Finally, return very short summary of filter interpretations on the
    # validation set.
    p_acts <- get_phrase_acts(model = model, params = these_params,
                              embeds = embeds[temp_dat$fold == "test", , ])
    grid$phrases[i] <- get_top_phrases_quick(phrase_acts = p_acts,
                                             tokens = tokens,
                                             params = these_params,
                                             vocab = vocab)

    # Free up memory (only partially successful in my past experience)
    rm(list = c("p_acts", "acts", "model"))
    keras::k_clear_session()
    tf$keras$backend$clear_session()
    gc()
  }

  return(grid)

}
