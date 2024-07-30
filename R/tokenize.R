#' Tokenize input text
#'
#' Input text is converted to tokens. This can be performed with one of the
#' following methods:
#'  1. A default BERT sub-word tokenizer (requires input for `max_length`)
#'  2. Chosen tokenizer model name from the Hugging Face transformer library
#'     (requires input for `tokenizer` and `max_length`)
#'  3. Existing tokens uploaded from a file (requires inputs for `file_name`
#'     and `token_map`). Order is assumed to match between `x` and the token
#'     data in `file_name`.
#'
#' @param x A data frame with a `text` column.
#' @param tokenizer A tokenizer model name from the Hugging Face transformer
#'        library.
#' @param max_length Max number of tokens to consider per text sample. Longer
#'        texts with tokens will be cut off beyond this threshold. Shorter
#'        values of `max_length` correspond to a faster model.
#' @param file_name Path to the file with tokenized text.
#' @param token_map Path to the file with the tokenizer vocabulary list (list
#'        names should be sub-words, values should be tokens).
#'
#' @return A list containing: a matrix with tokens corresponding to `x$text`,
#'         (`tokens`) and a list with the tokenizer vocabulary (`vocab`).
#'
#' @examples
tokenize <- function(x, tokenizer = NULL, max_length = NULL, file_name = NULL,
                     token_map = NULL) {
  ### TODO: This downloads files to the user's library -- this should be made
  ### clear.
  ### TODO: Keep in mind that vocab indexing starts from 0 with Python...
  model_name <- if (is.null(tokenizer)) "bert-base-uncased" else model_name
  if (is.null(file_name)) {
    if (is.null(max_length)) {
      stop("The variable max_length must be specified to perform tokenization.")
    }
    xfmr <- reticulate::import("transformers")
    tknzr <- xfmr$AutoTokenizer$from_pretrained(model_name)
    tokens <- tknzr(x$text, padding = "max_length", truncation = TRUE,
                    max_length = as.integer(max_length), return_tensors = "pt")
    vocab <- tknzr$get_vocab()
  } else {
    stop("This functionality still needs to be developed.")
    ### A potentially difficult aspect will be making sure the input matrix
    ### gets turned into a tensor with the proper format.
  }

  return(list("tokens" = tokens, "vocab" = vocab))
}
