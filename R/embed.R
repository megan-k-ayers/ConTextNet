#' Get embeddings of texts
#'
#' @param input_list This should be an output of prep_data().
#'
#' @return List, which is `input_list` now including the text embeddings, with
#' the embedding dimension recorded in `input_list$params$embed_dim`.
#'
#' @examples
#' \dontrun{
#' res <- embed(input_list)
#' }
embed <- function(input_list) {
  ### TODO: Is this the best place to use reticulate? Should this be in a setup
  ### script that saves it as a global variable?
  ### TODO: Eventually, examples should use intermediate files.
  if (input_list$embed_method != "file") {
    if (input_list$embed_method == "default") {
      model_name <- "prajjwal1/bert-tiny"
    } else if (input_list$embed_method == "name") {
      model_name <- input_list$embed_instr$name
    }
    xfmr <- reticulate::import("transformers")
    embed_model <- xfmr$AutoModel$from_pretrained(model_name)
    embeds <- embed_model(attention_mask = input_list$tokens$attention_mask,
                          input_ids = input_list$tokens$input_ids)
    embeds <- embeds$last_hidden_state
    embeds <- embeds$detach()$numpy()
    input_list$params$embed_dim <- dim(embeds)[3]
  } else {
    stop("This functionality still needs to be developed.")
  }
  ### TODO: Save in intermediate OUTPUT file which will get overwritten later.
  ### TODO: Have some text file which is a log of what has been completed
  ### so far. Would be useful especially when things get left on the cluster.
  input_list$embeds <- embeds
  return(input_list)
}
