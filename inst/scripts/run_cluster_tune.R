library(ConTextNet)
library(tensorflow)

# Handle command-line arguments for input file name, parameter grid row for
# this job to start at, and number of grid rows for this job.
args <- as.list(commandArgs(trailingOnly = TRUE))
names(args) <- c("input_file", "start_row", "batch_size")
args$start_row <- as.numeric(args$start_row)
args$batch_size <- as.numeric(args$batch_size)
args$end_row <- args$start_row + args$batch_size - 1

# Read in and unpack saved inputs.
input_embeds <- readRDS(args$input_file)
dat <- input_embeds$dat; embeds <- input_embeds$embeds
meta_params <- input_embeds$params; grid <- input_embeds$params$grid;
tokens <- input_embeds$tokens; vocab <- input_embeds$vocab
rm(list = "input_embeds"); gc()

if (nrow(grid) < args$end_row) args$end_row <- nrow(grid)

# Run tuning for this portion of the grid and save results.
tune_res <- tune_model(dat, embeds, meta_params,
                       grid[args$start_row:args$end_row, ], tokens, vocab)
saveRDS(tune_res, paste0("processed_data/", meta_params$folder,
                         "/tuning_results_", args$start_row,
                         "_", args$end_row, ".rds"))
