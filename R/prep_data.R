###############################################################################
###            PREPARE INPUT DATA FOR MODEL TRAINING AND ASSESSMENT.
###
### Runs:         Locally.
### Status:       Core functionality complete.
### Priority:     Medium.
### User facing:  Partially.
###############################################################################


#' Create new directory for model files
#'
#' This directory will contain the following sub directories...
#'
#' @param name Name to give the directory for this model
#'
#' @return The location of the directory created
#'
#' @examples
#' \dontrun{create_model_dir("example")}
create_model_dir <- function(name) {

  ### Check if name exists already

  ### If yes, do not create and throw a warning message

  ### If no, create it.

}


#' Scale covariates to have mean-zero and standard deviation of 1 using training
#' data.
#'
#' @param dat The input data set. Should include a `fold` column with `train`.
#'        and `test` entries.
#' @param vars The names of variables to scale.
#' @param scale_type Either "normalize" or "min-max".
#'
#' @return The same data frame, but with the `covs` columns scaled.
#' @export
#'
#' @examples
#' dat <- as.data.frame(matrix(rnorm(12), ncol = 3))
#' dat$fold <- "train"
#' dat2 <- scale_vars(dat, c("V1", "V2"), "min-max")
scale_vars <- function(dat, vars, scale_type) {

  ### Want to record the scaling parameters, so performing manually.
  dat_train <- dat[dat$fold == "train", vars, drop = FALSE]
  if (scale_type == "normalize") {
    means <- apply(dat_train, 2, mean)
    stdevs <- apply(dat_train, 2, stats::sd)
    dat[vars] <- sweep(sweep(dat[, vars, drop = FALSE], 2, means), 2, stdevs,
                         "/")
    res <- list("params" = list(means = means, stdevs = stdevs), "dat" = dat)

  } else if (scale_type == "min-max") {
    mins <- apply(dat_train, 2, min)
    maxs <- apply(dat_train, 2, max)
    for (i in 1:length(vars)) {
      v <- vars[i]
      dat[, v] <- -1 + (dat[, v] - mins[i])*2 / (maxs[i] -  mins[i])
    }
    res <- list("params" = list(mins = mins, maxs = maxs), "dat" = dat)
  } else {
    stop("Please specify an accepted method for scaling variables.")
  }
  return(res)
}


#' Create a list to store model parameters (meta-params and final model params)
#' Handles parameter setting, with different cases for parameter tuning and
#' direct specification (tuning = "none").
#' @param p A list, model_params, passed to prep_data().
#' @param tune_method How tuning should be performed: locally, via a Cluster
#'        with Slurm, generically via a shell script, or not at all
#'        ("local", "slurm", "shell", or "none")
#'
#' @return
#'
#' @examples
prep_params <- function(p, tune_method) {

  list_flag <- any(sapply(p, methods::is, class2 = "list"))
  if (list_flag) {  # If any list item is a list, check that its items vary.
    single_spec_flag <- identical(unique(sapply(p, length)), as.integer(1))
  } else single_spec_flag <- TRUE  # Otherwise we have a single specification.

  if (single_spec_flag) {
    if (list_flag) {  # Case with single specification but sub-lists --> unlist.
      params <- lapply(p, unlist)
    } else params <- p  # Case with single specification --> use as is.
  } else if (tune_method == "none") {  # Case with no tuning but >1 setting
    stop("Parameter settings are ill-specified - please review them.")
  } else {  # Case with tuning and >1 setting
    params <- list()  # Leave empty - will get only meta-params, others in grid.
  }
  return(params)
}


#' Build parameter grid for tuning
#'
#' @param model_params List defining the parameter values to consider during
#'        tuning.
#' @param K Number of folds (iterations) to train for each setting.
#' @param task Whether this is "reg" (regression) or "class" (classification).
#'
#' @return A tibble, where each row represents a different setting for model
#'         parameters.
#'
#' @examples
create_grid <- function(model_params, K, task) {

  grid <- tidyr::expand_grid("n_filts" = model_params$n_filts,
                             "kern_sizes" = model_params$kern_sizes,
                             "lr" = model_params$lr,
                             "lambda_cnn" = model_params$lambda_cnn,
                             "lambda_corr" = model_params$lambda_corr,
                             "lambda_out" = model_params$lambda_out,
                             "epochs" = model_params$epochs,
                             "batch_size" = model_params$batch_size,
                             "patience" = model_params$patience,
                             "covars" = model_params$covars)
  grid$id <- 1:nrow(grid)
  grid <- do.call("rbind", replicate(K, grid, simplify = FALSE))
  grid <- grid[order(grid$id), ]
  grid$run <- rep(1:K, times = length(unique(grid$id)))
  grid <- grid[, c("id", "run", setdiff(names(grid), c("id", "run")))]

  # Add space to record performance metrics/information
  if (task == "class") {
    grid$train_acc <- NA
    grid$train_f1 <- NA
    grid$val_acc <- NA
    grid$val_f1 <- NA
  }
  grid$train_mse <- NA
  grid$val_mse <- NA
  grid$act_range <- NA  # Size of filter activation ranges
  grid$max_corr <- NA   # Maximum corr between two filters' activations
  grid$phrases <- NA    # Summary of top phrases for each filter

  return(grid)
}


#' Prepare data for ConTextNet model
#'
#' @param x The input data as a data frame.
#' @param y_name The name of the outcome column `x`.
#' @param text_name The name of the text column in `x`.
#' @param model_params List defining the parameter values to consider during
#'        tuning.
#' @param task Whether this is "reg" (regression) or "class" (classification).
#' @param test_prop Proportion of `nrow(x)` to reserve in the test set.
#' @param embed_method How embedding should be performed ("file" to read from
#'        files, "name" for Hugging Face model referenced by name, or "default"
#'        for a default BERT model).
#' @param embed_instr Depending on choice for embed_method, a list containing
#'        The "file" method: the file path to read tokens from, the file path to
#'        read the token vocabulary list from, and the file path to read the
#'        token embeddings from (named "token_path", "vocab_path", and
#'        "embed_path")
#'        The "name" method: the name of the Hugging Face model to use and the
#'        max number of tokens to consider per text sample (named "name" and
#'        "max_length")
#'        The "default" method: the max number of tokens to consider per text
#'        sample (named "max_length")
#' @param tune_method How tuning should be performed: locally, via a Cluster
#'        with Slurm, generically via a shell script, or not at all
#'        ("local", "slurm", "shell", or "none")
#' @param folder_name Name of directory to create for saving model files.
#' @param folds Number of cross validation folds for tuning (default is NULL,
#'        must set to an integer if tune_method != "none").
#' @param scale_y Instructions for scaling the outcome. Default is "none." To
#'        scale, set to either "normalize" or "min-max".
#' @param scale_cov Instructions for scaling covariates. Default is "normalize".
#'        To scale, set to either "normalize" or "min-max". To avoid scaling,
#'        set to "none."
#'
#' @return
#' @export
#'
#' @examples \dontrun{
#' model_params <- list("n_filts" = list(4, 8),
#'                      "kern_sizes" = list(c(3, 5), c(3), c(5)),
#'                      "lr" = list(0.0001, 0.001),
#'                      "lambda_cnn" = list(0, 0.0001),
#'                      "lambda_corr" = 0, "lambda_out" = list(0, 0.0001),
#'                      "epochs" = 100, "batch_size" = 32,
#'                      "patience" = 20, "covars" = list(NULL))
#' res <- prep_data(x = imdb, y_name = "y", text_name = "text",
#'                  model_params = model_params, task = "class",
#'                  folder_name = "example", tune_method = "local", folds = 3)
#'
#' model_params <- list("n_filts" = 2, "kern_sizes" = c(3, 5), "lr" = 0.0001,
#'                    "lambda_cnn" = 0, "lambda_corr" = 0, "lambda_out" = 0,
#'                    "epochs" = 20, "batch_size" = 32, "patience" = 15,
#'                    "covars" = NULL)
#' res <- prep_data(x = imdb, y_name = "y", text_name = "text",
#'                  model_params = model_params, task = "class",
#'                  folder_name = "example")
#'
#' res <- prep_data(x = imdb, y_name = "y", text_name = "text",
#'                  model_params = model_params, task = "class",
#'                  embed_method = "name",
#'                  embed_instr = list("name" = "bert-base-cased",
#'                                     "max_length" = 200),
#'                  folder_name = "example")}
prep_data <- function(x, y_name, text_name,  model_params, task,
                      test_prop = 0.2, scale_y = "none", scale_cov = "normalize",
                      embed_method = "default",
                      embed_instr = list("max_length" = 200),
                      tune_method = "none", folder_name, folds = NULL) {

  ### Create directory for model files

  ### Rename outcome and text columns (if needed)

  ### Perform QA checks on data

  ### Error handling for task setting.
  if (task == "class" & length(unique(x$y)) > 2) {
    stop("Task is set to classification but more than 2 classes detected.")
  } else if (task == "reg" & length(unique(x$y)) == 2) {
    stop("Task is set to regression but only 2 unique outcomes detected. Did you mean to specify task = 'class'?")
  }

  ### Train/test split
  x$fold <- sample(c("train", "test"), nrow(x), replace = TRUE,
                   prob = c(1 - test_prop, test_prop))

  ### Prep formatting of meta-params (which will include model params if only a
  ### single model is being run).
  params <- prep_params(model_params, tune_method)
  params$n_tokens <- embed_instr$max_length
  params$folder <- folder_name
  params$task <- task

  ### Scale covariate columns (if included). The `scale_covars` function
  ### scales both the training and test sets using only the training data.
  if (!is.null(params$covars) & tune_method == "none" & scale_cov != "none") {
    x <- scale_vars(x, params$covars, scale_cov)$dat  # Case without tuning
  } else if (!is.null(unlist(model_params$covars)) & scale_cov != "none") {
    x <- scale_vars(x, unique(unlist(model_params$covars)),
                    scale_cov)$dat # Case with tuning
  }

  ### Similarly, scale the outcome if it is continuous.
  if (task == "reg" & scale_y != "none") x <- scale_vars(x, "y", scale_y)$dat


  ### Tokenize the text, maintain unique IDs?
  token_res <- tokenize(x, embed_method = embed_method,
                        embed_instr = embed_instr)


  ### Create parameter grid if tuning is happening
  if (tune_method != "none") {
    grid <- create_grid(model_params = model_params, K = folds, task = task)

    # Define folds within the training set for tuning cross-validation
    inds <- which(x$fold == "train")
    x$tune_fold <- NA
    x$tune_fold[inds] <- sample(cut(seq(1, length(inds)), breaks = folds,
                                    labels = FALSE))
  }

  ### Write shell script(s), whatever will be needed for running on the cluster,
  ### unless choosing to run locally.


  ### Build input file with everything necessary for the model
  input <- list(dat = x,
                params = params,
                tokens = token_res$tokens,
                token_mask = token_res$mask,
                vocab = token_res$vocab,
                embed_method = embed_method,
                embed_instr = embed_instr,
                tune_method = tune_method,
                grid = grid)

  ### See if you can zip up the input list and the cluster scripts?

  return(input)

}
