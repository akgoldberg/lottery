# Directory to reproduce Swiss NSF intervals using Bayesian hierarchical model

### Code from: https://snsf-data.github.io/ERpaper-online-supplement/#
### To use, need to download data individual_votex.xlsx from: https://zenodo.org/records/4531160 and put in this directory.

### To run using docker:
> docker build -t er-research .     
> docker run -it er-research
> Run R script reproduce_results.r from inside docker bash shell

# To use RStudio server 
> Run docker run -d --name er-research -e PASSWORD=yourpassword -p 8787:8787 er-research
> Visit localhost:8787