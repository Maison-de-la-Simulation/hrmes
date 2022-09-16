# Concept note on the use of IA to accelerate spinup of OGCM

## Motivation

* an illustration of spin-up time from the QUEST project

Illustration of spin-up of IPSLCM6.2.2-MR025 (orange) and bifurcation towards another equilibrium (purple) after ocean currents were (mistakingly) reset to 0 ; other lines illustrate the long-term drift of IPSLCM6.2.2-MR1 (green and blue) and the bifurcation due to change in CO2 concentration (yellow and red). Variable is the temperature at 1000m depth, globally averaged.
![Spinup of QUEST-MR025](data/OCE_thetao_1000m_MR025_pi_spinup.gif)

* another illustration 

![Set1 of simulations](data/CM62-CM65_OCE_hc.gif)


## Dataset


Tableau qui liste les 7 simulations disponibles pour training and testing database

Figure qui montre MSFT moyenne pour la simu la plus longue avec petits points + pannel qui montre series temporelles en qq points choisis -> montrer que signal varie n'importe comment un peu partout...

## First step : reducing complexity in datatset

* PCA
PCA sur 1 seule simu à la fois 
Figure qui montre series temporelles du 1er mode pour chaque simu + reconstruction du 1er mode à partir de PCA sur la plus longue

Figure qui montre carte du 1er mode sur serie la plus longue

Figure qui montre series temporelles PCA pour 2e mode pour chaque simu

Figure qui montre series temporelles pour 3e mode pour chaque simu

* DMD

????

## Second step : emulating spin up time series

Figure qui montre 1er mode avec gaussian guess...

## Third step : injecting accelerated MSFT into NEMO

Figure qui montre ?!?




