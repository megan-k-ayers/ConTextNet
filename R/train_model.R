###############################################################################
###                            TRAIN CONTEXTNET MODEL
###
### Runs:         On HPC cluster.
### Status:       Almost complete - mainly tests and documentation remaining.
### Priority:     Medium.
### User facing:  No.
###############################################################################


#' Train a ConTextNet Model
#'
#' @param dat Original text data set
#' @param embeds Embeddings corresponding to dat$text
#' @param params Model parameters
#' @param run_quiet Logical indicating whether or not to show model training
#'        progress in the console.
#'
#' @return Trained Keras model.
#' @export
#'
#' @examples \dontrun{train_model(imdb_embed)}
train_model <- function(dat, embeds, params, run_quiet = FALSE) {

  train_inds <- which(dat$fold == "train")
  x_train <- embeds[train_inds, , ]
  y_train <- dat$y[train_inds]
  cov_flag <- !is.null(params$covars)

  ### Calculate class weights for classification tasks.
  if (params$task == "class") {
    class_wts <- list("0" = (1 / sum(y_train == 0)) * (nrow(x_train) * 4 / 5),
                      "1" = (1 / sum(y_train == 1)) * (nrow(x_train) / 5))
  } else class_wts <- NULL

  ### Prep covariates as inputs, if using them.
  if (cov_flag) c_train <- as.matrix(dat[train_inds, params$covars])
  train_inputs <- if (cov_flag) list(x_train, c_train) else x_train

  ### Initialize the model.
  model <- init_model(params, train_inputs)

  ### Train the model with early stopping.
  callback = tf$keras$callbacks$EarlyStopping(monitor = 'val_loss',
                                              patience = params$patience)
  verbose <- if (run_quiet) 0 else getOption("keras.fit_verbose",
                                             default = "auto")
  history <- model %>%
    keras::fit(train_inputs, y_train, epochs = params$epochs,
               batch_size = params$batch_size,
               validation_split = 0.2,  # Using 20% of train set for validation.
               class_weight = class_wts,
               callbacks = list(callback),
               verbose = verbose)

  return(model)

}


#' Evaluate a Trained ConTextNet Model's Performance
#'
#' @param model A trained ConTextNet model, the output of train_model().
#' @param input_dat Input data to evaluate the model with. If covariates are
#'        considered by the model, this should be a list with two elements:
#'        the input embeddings and the input covariates.
#' @param y True labels corresponding to input_dat.
#' @param metrics Character array containing the metrics to evaluate the model
#'        with (options include "mse", "accuracy", "f1").
#'
#' @return Data frame summarizing model performance.
#' @export
#'
#' @examples \dontrun{
#' model <- train_model(imdb_embed)
#' input_dat <- imdb_embed$embeds[imdb_embed$dat$fold == "test", , ]
#' y <- imdb_embed$dat$y[imdb_embed$dat$fold == "test"]
#' metrics <- c("mse", "accuracy", "f1")
#' eval_model(model, input_dat, y, metrics)
#' }
eval_model <- function(model, input_dat, y, metrics) {
  preds <- stats::predict(model, input_dat, verbose = 0)
  res <- data.frame(metric = character(), value = numeric())
  if ("mse" %in% metrics) {
    res <- rbind(res, data.frame(metric = "mse", value = mean((preds - y)^2)))
  }
  if ("accuracy" %in% metrics) {
    res <- rbind(res, data.frame(metric = "accuracy",
                                 value = mean(round(preds) == y)))
  }
  if ("f1" %in% metrics) {
    res <- rbind(res, data.frame(metric = "f1",
                                 value = MLmetrics::F1_Score(y, round(preds))))
  }

  return(as.data.frame(tidyr::pivot_wider(res, names_from = "metric",
                                          values_from = "value")))

}
