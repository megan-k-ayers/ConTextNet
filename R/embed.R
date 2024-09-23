###############################################################################
###                            EMBED INPUT TOKENS
###
### Runs:         On HPC cluster.
### Status:       Core functionality complete.
### Priority:     Medium.
### User facing:  No.
###############################################################################


#' Get embeddings of texts
#'
#' @param input_list This should be an output of prep_data().
#'
#' @return List, which is `input_list` now including the text embeddings, with
#' the embedding dimension recorded in `input_list$params$embed_dim`.
#'
#' @examples
#' \dontrun{
#' res <- embed(imdb_input_list)
#' }
embed <- function(input_list) {

  if (input_list$embed_method != "file") {
    if (input_list$embed_method == "default") {
      model_name <- "prajjwal1/bert-tiny"
    } else if (input_list$embed_method == "name") {
      model_name <- input_list$embed_instr$name
    }
    xfmr <- reticulate::import("transformers")
    torch <- reticulate::import("torch")

    # Need to reformat tokens and attention mask as torch tensors for the
    # transformers library.
    tokens <- torch$tensor(input_list$tokens, dtype = torch$int64)
    mask <- torch$tensor(input_list$token_mask, dtype = torch$int64)

    # Get embeddings, store them back in the model input list along with the
    # embedding dimension.
    embed_model <- xfmr$AutoModel$from_pretrained(model_name)
    embeds <- embed_model(attention_mask = mask, input_ids = tokens)
    embeds <- embeds$last_hidden_state
    embeds <- embeds$detach()$numpy()
    input_list$params$embed_dim <- dim(embeds)[3]
  } else {
    stop("This functionality still needs to be developed.")
  }

  input_list$embeds <- embeds
  return(input_list)
}
