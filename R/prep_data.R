#' Create new directory for model files
#'
#' This directory will contain the following subdirectories
#'
#' @param name Name to give the directory for this model
#'
#' @return The location of the directory created
#'
#' @examples
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
#' @param grid_vals List defining the parameter values to consider during
#'        tuning.
#' @param task Whether this is "reg" (regression) or "class" (classification).
#' @param token_method How tokenizing should be performed ("file" for embeddings
#'        read from a file, "name" for Hugging Face model referenced by name,
#'        "default" for default BERT sub-word tokenization).
#' @param max_length Max number of tokens to consider per text sample. Longer
#'        texts with tokens will be cut off beyond this threshold. Shorter
#'        values of `max_length` correspond to a faster model.
#' @param token_instr Either a list with (1) the relative file name to read
#'        tokens from and (2) the relative file name to read the vocabulary
#'        list from, or the name of the Hugging Face model to use.
#' @param embed_method How embedding should be performed ("file" for embeddings
#'        read from a file, "name" for Hugging Face model referenced by name).
#' @param embed_instr Either the relative file name to read embeddings from, or
#'        the name of the Hugging Face model to use.
#' @param tune_method How tuning should be performed: locally, via a Cluster
#'        with Slurm, or generically via a shell script.
#'
#' @return
#' @export
#'
#' @examples
#' \dontrun{
#' res <- prep_data(imdb, "y", "text", list(), "class")
#' res <- prep_data(imdb, "y", "text", list(), "class", token_method = "name",
#'                  token_instr = "bert-base-cased")}
prep_data <- function(x, y_name, text_name, grid_vals, task,
                      token_method = "default", max_length = 200,
                      token_instr = NULL, embed_method = "default",
                      embed_instr = NULL, tune_method = "local") {

  ### Create directory for model files

  ### Rename outcome and text columns (if needed)

  ### Perform QA checks on data

  ### Unique IDs to samples

  ### Tokenize the text, maintain unique IDs?
  if (token_method == "default") {
    token_res <- tokenize(x, max_length = max_length)
  } else if (token_method == "file") {
    token_res <- tokenize(x, file_name = token_instr[[1]],
                                      token_map = token_instr[[2]])
  } else if (token_method == "name") {
    token_res <- tokenize(x, tokenizer = token_instr, max_length = max_length)
  } else {
    stop("Please input a valid option for token_method.")
  }


  ### Write token map (stays local)

  ### Train/test split

  ### Create parameter grid (unless grid_vals is list of vectors of length 1)

  ### Create params list

  ### Write shell script(s), whatever will be needed for running on the cluster,
  ### unless choosing to run locally.

  ### Build input file with everything necessary for the model
  input <- list(dat = x,
                params = list(),
                tokens = token_res$tokens,
                vocab = token_res$vocab,
                embed_method = embed_method,
                embed_instr = embed_instr,
                tune_method = tune_method,
                grid = data.frame())

  ### See if you can zip up the input list and the cluster scripts?

  return(input)

}
