#' Create new directory for model files
#'
#' This directory will contain the following subdirectories
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


#' Create a list to store model parameters (meta-params and final model params)
#'
#' @param p A list, model_params, passed to prep_data().
#' @param tune_method How tuning should be performed: locally, via a Cluster
#'        with Slurm, generically via a shell script, or not at all
#'        ("local", "slurm", "shell", or "none")
#'
#' @return
#'
#' @examples
prep_params <- function(p, tune_method) {
  ### Handle parameters, with different cases for parameter tuning and direct
  ### specification (tuning = "none").
  ### TODO: Handling if user tries to specify parameters for tuning using c()
  ### instead of list(). This assumes ~perfect behavior (user gives sub-lists
  ### only if parameter tuning or passing single specification, otherwise gives
  ### specification single directly.
  list_flag <- any(sapply(p, methods::is, class2 = "list"))
  if (list_flag) {
    single_spec_flag <- identical(unique(sapply(p, length)), as.integer(1))
  } else single_spec_flag <- TRUE

  if (single_spec_flag) {
    if (list_flag) {  # Case with single specification but sub-lists
      params <- lapply(p, unlist)
    } else params <- p  # Case with single specification
  } else if (tune_method == "none") {  # Case with no tuning but >1 setting
    stop("Parameter settings are ill-specified - please review them.")
  } else {  # Case with tuning and >1 setting
    params <- list()
  }
  return(params)
}


#' Build parameter grid for tuning
#'
#' @return
#'
#' @examples
create_grid <- function() {
  return(1)
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
#'
#' @return
#' @export
#'
#' @examples
#' model_params <- list("n_filts" = list(2), "kern_sizes" = list(c(3, 5)),
#'                    "lr" = list(0.0001), "lambda_cnn" = list(0),
#'                    "lambda_corr" = list(0), "lambda_out" = list(0),
#'                    "epochs" = list(20), "batch_size" = list(32),
#'                    "patience" = 15, "covars" = list(NULL))
#' res <- prep_data(x = imdb, y_name = "y", text_name = "text",
#'                  model_params = model_params, task = "class",
#'                  folder_name = "example")
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
#'                  folder_name = "example")
prep_data <- function(x, y_name, text_name, model_params, task, test_prop = 0.2,
                      embed_method = "default",
                      embed_instr = list("max_length" = 200),
                      tune_method = "none", folder_name) {

  ### Create directory for model files

  ### Rename outcome and text columns (if needed)

  ### Perform QA checks on data

  ### Unique IDs to samples

  ### Tokenize the text, maintain unique IDs?
  token_res <- tokenize(x, embed_method = embed_method,
                        embed_instr = embed_instr)

  ### Write token map (stays local)

  ### Train/test split
  ### TODO: Non-invasive set.seed here. And make it exact to test_prop. And
  ### move to a helper function?
  x$fold <- sample(c("train", "test"), nrow(x), replace = TRUE,
                   prob = c(1 - test_prop, test_prop))

  ### Prep formatting of meta-params (which will include model params if only a
  ### single model is being run).
  params <- prep_params(model_params, tune_method)
  params$n_tokens <- embed_instr$max_length
  params$folder <- folder_name
  params$task <- task

  ### Scale covariate columns (if included)
  if (!is.null(params$covars)) {  # Case without tuning
    x <- scale(x[, params$covars])
  } else if (any(!is.null(model_params$covars))) {  # Case with tuning
    x <- scale(x[, unique(unlist(model_params$covars))])
  }

  ### Create parameter grid if tuning is happening


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
                grid = data.frame())

  ### See if you can zip up the input list and the cluster scripts?

  return(input)

}
