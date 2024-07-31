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
#' @return
#'
#' @examples
create_params <- function() {
  return(1)
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
#' @param param_vals List defining the parameter values to consider during
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
#' \dontrun{
#' param_vals <- list("n_filts" = list(2), "kern_sizes" = list(c(3, 5)),
#'                    "lr" = list(0.0001), "lambda_cnn" = list(0),
#'                    "lambda_corr" = list(0), "lambda_out" = list(0),
#'                    "epochs" = list(20), "batch_size" = list(32),
#'                    "covars" = list(NULL))
#' res <- prep_data(x = imdb, y_name = "y", text_name = "text",
#'                  param_vals = param_vals, task = "class",
#'                  folder_name = "example")
#'
#' res <- prep_data(x = imdb, y_name = "y", text_name = "text",
#'                  param_vals = param_vals, task = "class",
#'                  embed_method = "name",
#'                  embed_instr = list("name" = "bert-base-cased",
#'                                     "max_length" = 200),
#'                  folder_name = "example")
#' }
prep_data <- function(x, y_name, text_name, param_vals, task, test_prop = 0.2,
                      embed_method = "default",
                      embed_instr = list("max_length" = 200),
                      tune_method = "none", folder_name) {

  ### Create directory for model files

  ### Rename outcome and text columns (if needed)

  ### Perform QA checks on data

  ### Unique IDs to samples

  ### Tokenize the text, maintain unique IDs?
  if (embed_method == "default") {
    token_res <- tokenize(x, max_length = embed_instr$max_length)
  } else if (embed_method == "file") {
    token_res <- tokenize(x, token_path = embed_instr$token_path,
                          vocab_path = embed_instr$vocab_path)
  } else if (embed_method == "name") {
    token_res <- tokenize(x, tokenizer = embed_instr$name,
                          max_length = embed_instr$max_length)
  } else {
    stop("Please input a valid option for token_method.")
  }

  ### Write token map (stays local)

  ### Train/test split
  ### TODO: Non-invasive set.seed here. And make it exact to test_prop. And
  ### move to a helper function?
  x$fold <- sample(c("train", "test"), nrow(x), replace = TRUE,
                   prob = c(1 - test_prop, test_prop))

  ### Separate tuned parameters from fixed parameters, prepare for tuning (if
  ### happening).
  if (tune_method == "none") {
    if (!identical(unique(sapply(param_vals, length)), as.integer(1))) {
      stop("Parameter settings are ill-specified - please review them.")
    }
    params <- param_vals
    params$n_tokens <- embed_instr$max_length
    params$folder = folder_name
    params$covars = param_vals$covars
  } else {
    params <- list("n_tokens" = embed_instr$max_length,
                   "folder" = folder_name,
                   "covars" = param_vals$covars)

    ### Create parameter grid (unless param_vals is list of vectors of length 1)


    ### Write shell script(s), whatever will be needed for running on the cluster,
    ### unless choosing to run locally.


  }


  ### Build input file with everything necessary for the model
  input <- list(dat = x,
                params = param_vals,
                tokens = token_res$tokens,
                vocab = token_res$vocab,
                embed_method = embed_method,
                embed_instr = embed_instr,
                tune_method = tune_method,
                grid = data.frame())

  ### See if you can zip up the input list and the cluster scripts?

  return(input)

}
