#' Get embeddings of texts
#'
#' @param input_list This should be an output of prep_data().
#'
#' @return
#'
#' @examples
#' \dontrun{
#' input_list <- prep_data(imdb, "y", "text", list(), "class")
#' res <- embed(input_list)}
#'
#' \dontrun{
#' input_list <- prep_data(imdb, "y", "text", list(), "class", embed_method = "name",
#'                  embed_instr = list("name" = "bert-base-cased",
#'                                     "max_length" = 200))
#' res <- embed(input_list)}
#'
embed <- function(input_list) {
  ### TODO: Is this the best place to use reticulate? Should this be in a setup
  ### script that saves it as a global variable?
  if (input_list$embed_method != "file") {
    if (input_list$embed_method =="default") {
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
  } else {
    stop("This functionality still needs to be developed.")
  }
  ### TODO: Save in intermediate OUTPUT file which will get overwritten later.
  ### TODO: Have some text file which is a log of what has been completed
  ### so far. Would be useful especially when things get left on the cluster.
  return(embeds)
}
