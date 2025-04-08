library(ERforResearch)
library(bayesplot)
source("swiss_nsf_modified.r")

path_to_xlsx <- "individual_votes.xlsx"

hss_s_mat <- openxlsx::read.xlsx(xlsxFile = path_to_xlsx, sheet = "pm_hsss") 
hss_s <- hss_s_mat %>% 
  get_right_data_format(prefix_assessor = "voter")

hss_h_mat <- openxlsx::read.xlsx(xlsxFile = path_to_xlsx, sheet = "pm_hssh") 
hss_h <- hss_h_mat %>% 
  get_right_data_format(prefix_assessor = "voter")

ls_b_mat <- openxlsx::read.xlsx(xlsxFile = path_to_xlsx, sheet = "pm_lsb") 
ls_b <- ls_b_mat %>% 
  get_right_data_format(prefix_assessor = "voter")

ls_m_mat <- openxlsx::read.xlsx(xlsxFile = path_to_xlsx, sheet = "pm_lsm") 
ls_m <- ls_m_mat %>% 
  get_right_data_format(prefix_assessor = "voter")

stem_mat <- openxlsx::read.xlsx(xlsxFile = path_to_xlsx, sheet = "pm_stem")
stem <- stem_mat %>% 
  get_right_data_format(prefix_assessor = "voter")

mint_section1_mat <- 
  openxlsx::read.xlsx(xlsxFile = path_to_xlsx, sheet = "mint_section1") 
mint_section1 <- mint_section1_mat %>% 
  get_right_data_format(prefix_assessor = "voter") %>% 
  mutate(section = "one")

mint_section2_mat <- 
  openxlsx::read.xlsx(xlsxFile = path_to_xlsx, sheet = "mint_section2") 
mint_section2 <- mint_section2_mat %>% 
  get_right_data_format(prefix_assessor = "voter") %>% 
  mutate(section = "two")

mint_section3_mat <- 
  openxlsx::read.xlsx(xlsxFile = path_to_xlsx, sheet = "mint_section3")
mint_section3 <- mint_section3_mat %>% 
  get_right_data_format(prefix_assessor = "voter") %>% 
  mutate(section = "three")

mint_section4_mat <- 
  openxlsx::read.xlsx(xlsxFile = path_to_xlsx, sheet = "mint_section4") 
mint_section4 <- mint_section4_mat %>% 
  get_right_data_format(prefix_assessor = "voter") %>% 
  mutate(section = "four")

# Note that we want to be able to differentiate the proposals in the different
# mint sections 
mint_sections <- mint_section1 %>% 
  mutate(proposal = paste0(proposal, "_1")) %>% 
  bind_rows(mint_section2%>% 
              mutate(proposal = paste0(proposal, "_2"))) %>% 
  bind_rows(mint_section3%>% 
              mutate(proposal = paste0(proposal, "_3"))) %>% 
  bind_rows(mint_section4%>% 
              mutate(proposal = paste0(proposal, "_4")))

##### How many projects can still be funded? ####
how_many_fundable <- c(7, 4, 7, 8, 6)
names(how_many_fundable) <- c("hss_s", "hss_h", "ls_m", "ls_b", "stem")


#### JAGS model for continuous outcome ####
# default model
n.iter <- 400 * 10^{3} / 2.
n.burnin <- 150 * 10^{3} / 2.
n.adapt <- 100 * 10^{3} / 2.
n.chains <- 4
seed <- 1991

# if mcmc_mint_object.RData already exists, load it
if (file.exists("mcmc_mint_object.RData")) {
  # load the object
  print("Loading mcmc_mint_object.RData")
  load("mcmc_mint_object.RData")
} else {
  # if it does not exist, run the function
  mcmc_mint_object <- 
    print("Running get_mcmc_samples()")
    get_mcmc_samples(data = mint_sections,
                     id_proposal = "proposal",
                     id_assessor = "assessor",
                     id_panel =  "section",
                     grade_variable = "num_grade",
                     n_chains = n.chains, n_iter = n.iter,
                     n_burnin = n.burnin, n_adapt = n.adapt,
                     max_iter = 400 * 10^{3} / 2.,
                     inits_type = "random",
                     rhat_threshold = 1.1,
                     seed = seed, quiet = FALSE,
                     dont_bind = TRUE)
  save(mcmc_mint_object, file = "mcmc_mint_object.RData")
}

mcmc_samples <- do.call(rbind, mcmc_mint_object$samples$mcmc)

plot_er_cred_mint <- plot_er_distributions(mcmc_mint_object, 
                      n_proposals = mint_sections %>%
                        summarise(n = n_distinct(proposal)) %>% pull(), 
                      number_fundable = 106)

zoomtheme <- theme(legend.position = "none", axis.line = element_blank(),
                   axis.text.x = element_blank(), axis.text.y = element_blank(),
                   axis.ticks = element_blank(), axis.title.x = element_blank(),
                   axis.title.y = element_blank(), title = element_blank(),
                   panel.grid.major = element_blank(),
                   panel.grid.minor = element_blank(),
                   panel.background = element_rect(color = 'red',
                                                   fill = "white"))

p.zoomrast <- plot_er_cred_mint + 
  xlim(levels(plot_er_cred_mint$data$parameter)[90:121]) +
  ylim(c(74, 140)) + 
  zoomtheme

plot_er_cred_mint + 
  theme(legend.position = "none") +
  annotation_custom(grob = ggplotGrob(p.zoomrast),
                    xmin = 160, xmax = 360, 
                    ymin = 5,
                    ymax = 100)

# Save the plot
ggsave("mint_plot.png", plot = last_plot(), 
       device = "png", width = 8, height = 4, dpi = 300)


### Get intervals ###
n_proposals = mint_sections %>%
  summarise(n = n_distinct(proposal)) %>% pull()
name_er_or_theta = "rank_theta"
er_colnames <- paste0(name_er_or_theta, "[", seq_len(n_proposals), "]")
er_samples <-mcmc_samples[, er_colnames]

inner_credible <- 0.5
outer_credible <- 0.9
mcmc_intervals_data_mat <- mcmc_intervals_data(er_samples, point_est = "mean",
                                                 prob = inner_credible,
                                                 prob_outer = outer_credible)

write.csv(mcmc_intervals_data_mat, "intervals.csv", row.names = FALSE)