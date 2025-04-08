library("ERforResearch")
library("bayesplot")

# Define function that makes the funding decisions based on the estimated expected rank
make_decisions <- function(mcmc_samples_object, n_proposals, n_chosen, inner_credible=0.5, outer_credible=0.95) {
    er_colnames <- paste0("rank_theta", "[", seq_len(n_proposals), "]")
    if (is.list(mcmc_samples_object$samples)) {
        er_samples <- do.call(rbind, mcmc_samples_object$samples$mcmc)[, er_colnames]
    } else {
        er_samples <- mcmc_samples_object$samples[, er_colnames]
    }

    # get credible intervals for expeted rank that are used to make decisions
    mcmc_intervals_data_mat <- mcmc_intervals_data(er_samples, point_est = "mean",
                                                 prob = inner_credible,
                                                 prob_outer = outer_credible)

    order <- mcmc_intervals_data_mat %>%
        arrange(.data$m) %>%
        dplyr::pull(.data$parameter)

    # If we draw a funding line, the provisional funding line is equal to the
    # ER of the last fundable proposal
    line_at <- mcmc_intervals_data_mat %>%
      arrange(.data$m) %>%
      slice(n_chosen) %>%
      dplyr::pull(.data$m)

    # The names of the different recommendation groups; by default accepted,
    # rejected, and random selection groups.
    group_names <- c(accepted = "accepted",
                     rejected = "rejected",
                     "random selection" = "random selection")
    # Distribute into the different recommendation groups, depending on whether
    # the decision should be taken based on the inner or on the outer CrI.
    mcmc_intervals_data_mat <- mcmc_intervals_data_mat %>%
        mutate(rs = ifelse(((.data$l <= line_at) &
                             (.data$h >= line_at)) |
                             .data$m == line_at,
                           # To ensure the special case where the ER of the last
                           # fundable is not in the CrI of the last fundable,
                           # because of the ER being the mean and the CrI
                           # quantiles.
                           group_names["random selection"],
                           ifelse((.data$l < line_at) &
                                    (.data$h < line_at),
                                  group_names["accepted"], group_names["rejected"])),
               decision = factor(.data$rs, group_names))
    
    return (mcmc_intervals_data_mat[, c("parameter", "l", "m", "h", "decision")])
}

# Define function that takes command line argument of data file, loads data and runs ER
run <- function(data_file, n_chosen = 1) {
    # if (data_file == "no_data") {
        # print("Using mock data")
        data <- get_mock_data() %>% 
            filter(panel == "p1") %>%
            head(10)
    # } else {
    #     print(paste("Using data from", data_file))
    #     data <- read.csv(data_file)
    # }
    # Estimate theta's
    mcmc_samples_object <- 
    get_mcmc_samples(
        data = data, 
        id_proposal = "proposal",
        id_assessor = "assessor",
        grade_variable = "num_grade",
        path_to_jags_model = NULL, 
        # NULL means we use the default model
        seed = 6,
        rhat_threshold = 1.05,
        dont_bind = TRUE)
    
    # Make decisions
    n_proposals <- data %>%
                        summarise(n = n_distinct(proposal)) %>%
                        pull()
    decisions <- make_decisions(mcmc_samples_object, n_proposals, n_chosen)

    return (decisions)
}

# Run the function
run(commandArgs(trailingOnly = TRUE))