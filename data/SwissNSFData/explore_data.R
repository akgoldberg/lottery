library(ERforResearch)

path_to_xlsx <- "individual_votes.xlsx"

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

# Get sparsity of each matrix
sparsity_mint_section1 <- sum(is.na(mint_section1_mat)) 
sparsity_mint_section2 <- sum(is.na(mint_section2_mat)) 
sparsity_mint_section3 <- sum(is.na(mint_section3_mat)) 
sparsity_mint_section4 <- sum(is.na(mint_section4_mat)) 
# Get total nubmer of entires in each matrix
total_entries_mint_section1 <- nrow(mint_section1_mat) * ncol(mint_section1_mat)
total_entries_mint_section2 <- nrow(mint_section2_mat) * ncol(mint_section2_mat)
total_entries_mint_section3 <- nrow(mint_section3_mat) * ncol(mint_section3_mat)
total_entries_mint_section4 <- nrow(mint_section4_mat) * ncol(mint_section4_mat)

total_entries <- total_entries_mint_section1 + total_entries_mint_section2 + 
  total_entries_mint_section3 + total_entries_mint_section4
sparsity <- sparsity_mint_section1 + sparsity_mint_section2 + 
  sparsity_mint_section3 + sparsity_mint_section4
sparsity_ratio <- sparsity / total_entries
print(paste0("Sparsity ratio: ", sparsity_ratio))


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

print(mint_sections)


