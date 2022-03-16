## 9 mars

* premier reseau CNN entrainé par Nathan donne des resultats tres prometteurs meme si l’amplitude du signal reconstruit est moindre (voir message suivant avec figure)
* Nathan travaille sur pytorch, et cela ravit Redouane :wink:
* normalisation point par point, avec min et max -> envisager de passer à ÷ std ou ÷(2×std)
* masque le long des continents à paufiner
* introduire periodicité entre bords est et ouest via padding (il y a aussi une periodicite le long du bord nord, entre parties est et ouest, mais plus difficile à introduire)

Prochaine étape : appliquer le reseau en boucle pour voir au bout de combien d’annees les erreurs deviennent trop importantes (tracer rmse en fonction du nombre de mois predits dans le futur, puis qq cartes pour points d’inflexion dans serie temporelle de rmse)

![First attempt with CNN](data/attempt1.png)
À gauche le 13ème mois prédit par le réseau à partir des 12 premiers, à gauche la carte attendue

## 16 février

* attention à la normalisation : il y a des liens etnre les differents pas de temps qu’il faut conserver
* questions autour de l’echantillonage pour eviter cycle saisonnier (on travaille sur tous les mois de decembre ? ou sur les moyennes DJF ? ou sur les moyennes annuelles ?)
* commencer par se familiariser avec les donnees en developpant un reseau tres simple (pb des donnees masquees…)
* prochaine reunion mercredi 9 mars à 10h00 (on choisira ensuite un autre creneau regulier plus compatible avec l’horaire de Balaji, qui sera de retour aux USA)

Exemples donnés par Julie pour enlever le cycle saisonnier, etc: 

la premiere solution (on enleve moyenne de tous les mois de janvier, fevrier, mars… ligne bleue) conserve exactement le spectre de reference (ligne noire) sauf les pics annuel et bi-annuel, tandis que les 2 autres solutions (Decembre seulement, ligne rouge, et moyenne DJF, ligne rose) ont plus de variance aux periodes < 16 ans, mais c’est à peine significatif. 
Entre Decembre et DJF, il y a plus de variance pour Decembre aux periodes < 10 ans, comme attendu par Balaji.
Conclusion : Si on considère que notre objectif est d’etre au plus proche de la reference, alors il vaut mieux prendre ssca, mais alors on doit travailler avec donnees mensuelles, et sinon DJF (donnees annuelles, donc 12 fois moins de points), et en dernier choix Decembre.