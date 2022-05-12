# Concept note on the use of IA to accelerate spinup of OGCM

## Motivation

* some illustrations of spin-up time
* Nathan travaille sur pytorch, et cela ravit Redouane :wink:
* normalisation point par point, avec min et max -> envisager de passer à ÷ std ou ÷(2×std)
* masque le long des continents à paufiner
* introduire periodicité entre bords est et ouest via padding (il y a aussi une periodicite le long du bord nord, entre parties est et ouest, mais plus difficile à introduire)

Prochaine étape : appliquer le reseau en boucle pour voir au bout de combien d’annees les erreurs deviennent trop importantes (tracer rmse en fonction du nombre de mois predits dans le futur, puis qq cartes pour points d’inflexion dans serie temporelle de rmse)

![First attempt with CNN](data/attempt1.png)
À gauche le 13ème mois prédit par le réseau à partir des 12 premiers, à gauche la carte attendue

