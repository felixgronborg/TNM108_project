import csv
import pandas as pd 

# Verkar fungera som det ska förutom 2 tweets från Trump som vi får ta bort
trump_df = pd.read_csv("trump.csv", error_bad_lines=False, sep='\t')
obama_df = pd.read_csv("obama.csv", error_bad_lines=False, sep=',')
