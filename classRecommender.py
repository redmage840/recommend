import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
%matplotlib inline
#line-by-line runtime comparison for easier code optimization.
# %load_ext line_profiler

pd.set_option('display.max_rows',1000)

elite = pd.read_csv('../inputs/boardgame-elite-users.csv')
elite = elite.rename(columns = {'Compiled from boardgamegeek.com by Matt Borthwick':'UserID'})
titles = pd.read_csv('../inputs/boardgame-titles.csv')
titles = titles.rename(columns={'boardgamegeek.com game ID':'gameID'})
frequent = pd.read_csv('../inputs/boardgame-frequent-users.csv')
frequent = frequent.rename(columns = {'Compiled from boardgamegeek.com by Matt Borthwick':'UserID'})