#' Plot Outcome vs. Document-Level Activations
#'
#' @param doc_acts Data frame with document-level activations, the output of
#'        get_doc_acts().
#'
#' @return A `ggplot` scatterplot with a facet per convolutional filter.
#'
#' @examples \dontrun{
#' imdb_embed$params$epochs <- 100
#' model <- train_model(imdb_embed)
#' embeds <- imdb_embed$embeds[imdb_embed$dat$fold == "train", , ]
#' dat <- imdb_embed$dat[imdb_embed$dat$fold == "train", ]
#' params <- imdb_embed$params
#' plot_doc_acts(get_doc_acts(model, embeds, params, dat))
#' }
plot_doc_acts <- function(doc_acts) {

  doc_acts$plot_col <- ifelse(doc_acts$wt > 0, "positive", "negative")

  ### Scatter plot with facets for each filter.
  gg <- ggplot2::ggplot(doc_acts,
                        ggplot2::aes(x = activation, y = y, color = plot_col))
  if (length(unique(doc_acts$y)) <= 2) {
    gg <- gg + ggplot2::geom_jitter(alpha = 0.5, height = 0.25)
  } else {
    gg <- gg + ggplot2::geom_point(ggplot2::aes(x = activation, y = y),
                                   alpha = 0.5)
  }

  gg <- gg + ggplot2::geom_smooth(method = "lm", color = "gray30") +
    ggplot2::facet_wrap(ggplot2::vars(filter))

  return(gg)

}
