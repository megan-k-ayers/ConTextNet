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
#'
#' @return A list containing: a matrix with tokens corresponding to `x$text`,
#'         (`tokens`) and a list with the tokenizer vocabulary (`vocab`).
#'
#' @examples
#' \dontrun{res <- tokenize(imdb)}
tokenize <- function(x, embed_method = "default",
                     embed_instr = list("max_length" = 200)) {
  ### TODO: Using default or name methods download files to the user's library
  ### -- this should be made clear.
  ### TODO: Keep in mind that vocab indexing starts from 0 with Python...
  ### TODO: Figure out better reticulate import solution. Should this be in a
  ### setup script that saves it as a global variable?
  ### TODO: Include default tokenizer/embedding model as a saved file rather
  ### than pulling it from Hugging Face.

  if (embed_method == "file") {
    stop("This functionality still needs to be developed.")

    ### A potentially difficult part will be making sure the input matrix
    ### gets turned into a tensor with the proper format.
    return(1)
  } else if (embed_method == "default") {
    model_name <- "prajjwal1/bert-tiny"
  } else if (embed_method == "name") {
    model_name <- embed_instr$name
  } else stop("Please input a valid option for token_method.")

  max_length <- embed_instr$max_length
  if (!methods::is(max_length, "numeric") | length(max_length) != 1) {
    stop("max_length must be specified as an integer in embed_instr.")
  }

  # Fast tokenizers often throw excessive parallelization warnings, so opting
  # not to use them for now.
  xfmr <- reticulate::import("transformers")
  tknzr <- xfmr$AutoTokenizer$from_pretrained(model_name, use_fast = FALSE,
                                              from_tf = TRUE)
  tokens <- tknzr(x$text, padding = "max_length", truncation = TRUE,
                  max_length = as.integer(max_length), return_tensors = "pt")
  vocab <- tknzr$get_vocab()
  vocab <- names(vocab)  # Just making vocab more intuitive to work with.

  return(list("tokens" = tokens$input_ids$numpy(),
              "mask" = tokens$attention_mask$numpy(),
              "vocab" = vocab))
}
