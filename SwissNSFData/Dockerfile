# Use the R + tidyverse + devtools base image
FROM rocker/verse:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    jags \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install required R packages
RUN R -e "install.packages(c('remotes', 'here', 'ggalluvial', 'rjags', 'runjags', 'openxlsx', 'psych', 'cowplot', 'gridExtra', 'lme4'))"
RUN R -e "remotes::install_github('snsf-data/ERforResearch')"

# Set memory limits (optional)
RUN echo "R_MAX_VSIZE=32Gb" >> /usr/local/lib/R/etc/Renviron \
 && echo "R_MAX_MEM_SIZE=32Gb" >> /usr/local/lib/R/etc/Renviron

# Copy your project into the container and run terminal to run R scripts
# COPY . /home/app/SwissNSFData
# WORKDIR /home/app/SwissNSFData
# CMD ["/bin/bash"]

# Comment the abvoe and uncomment the following lines to run RStudio Server
# Alternatively, you can start RStudio Server (Uncomment to run)
COPY . /home/rstudio/SwissNSFData
WORKDIR /home/rstudio/SwissNSFData
RUN chown -R rstudio:rstudio /home/rstudio/SwissNSFData
