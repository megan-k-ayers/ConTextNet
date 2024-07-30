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
#' @param x The input data as a data frame
#' @param y_name The name of the outcome column `x`
#' @param text_name The name of the text column in `x`
#' @param n_tokens Number of tokens to consider in column `text_name`
#' @param grid_vals List defining the parameter values to consider during tuning
#' @param task Whether this is "reg" (regression) or "class" (classification)
#' @param token_method How tokenizing should be performed ("file" for embeddings
#'        read from a file, "name" for Hugging Face model referenced by name)
#' @param token_instr Either the relative file name to read tokens from, or
#'        the name of the Hugging Face model to use
#' @param embed_method How embedding should be performed ("file" for embeddings
#'        read from a file, "name" for Hugging Face model referenced by name)
#' @param embed_instr Either the relative file name to read embeddings from, or
#'        the name of the Hugging Face model to use
#' @param tune_method How tuning should be performed: locally, via a Cluster
#'        with Slurm, or generically via a shell script.
#'
#' @return
#' @export
#'
#' @examples
prep_data <- function(x, y_name, text_name, n_tokens, grid_vals, task,
                      token_method, token_instr, embed_method, embed_instr,
                      tune_method = "local") {

  ### Create directory for model files

  ### Rename outcome and text columns

  ### Perform QA checks on data

  ### Unique IDs to samples

  ### Tokenize the text, maintain unique IDs

  ### Write token map (stays local)

  ### Train/test split

  ### Create parameter grid (unless grid_vals is list of vectors of length 1)

  ### Create params list

  ### Write shell script(s), whatever will be needed for running on the cluster,
  ### unless choosing to run locally.

  ### Build input file with everything necessary for the model
  input <- list(dat = data.frame(),
                params = list(),
                tokens = matrix(),
                embed_method = character(),
                embed_instr = character(),
                tune_method = character(),
                grid = data.frame())

  ### See if you can zip up the input list and the cluster scripts?

  return()

}
