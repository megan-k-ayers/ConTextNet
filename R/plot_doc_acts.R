#' Plot Outcome vs. Document-Level Activations
#'
#' @param doc_acts Data frame with document-level activations, the output of
#'        get_doc_acts().
#'
#' @return A `ggplot` scatterplot with a facet per convolutional filter.
#' @importFrom ggplot2 ggplot aes geom_jitter geom_point geom_smooth
#'
#' @examples \dontrun{
#' imdb_embed$params$epochs <- 100
#' model <- train_model(imdb_embed$dat, imdb_embed$embeds, imdb_embed$params)
#' embeds <- imdb_embed$embeds[imdb_embed$dat$fold == "train", , ]
#' dat <- imdb_embed$dat[imdb_embed$dat$fold == "train", ]
#' params <- imdb_embed$params
#' plot_doc_acts(get_doc_acts(model, embeds, params, dat))
#' }
plot_doc_acts <- function(doc_acts) {
  ### TODO: Add top 3 phrases in title? Definitely at least add output layer
  ### weight.

  doc_acts$plot_col <- ifelse(doc_acts$wt > 0, "positive", "negative")

  ### Scatter plot with facets for each filter.
  gg <- ggplot(doc_acts,
               aes(x = .data$activation, y = .data$y, color = .data$plot_col))
  if (length(unique(doc_acts$y)) <= 2) {
    gg <- gg + geom_jitter(alpha = 0.5, height = 0.25)
  } else {
    gg <- gg + geom_point(aes(x = .data$activation, y = .data$y), alpha = 0.5)
  }

  gg <- gg + geom_smooth(method = "lm", color = "gray30") +
    ggplot2::facet_wrap(ggplot2::vars(.data$filter))

  return(gg)

}
