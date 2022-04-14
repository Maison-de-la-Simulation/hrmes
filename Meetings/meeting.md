## 7 avril

* Redouane a refait tourner le notebook de Nathan, qui est donc reproductible et introduit qq modifications -> à partager avec Nathan
* test avec plus d’epochs -> donne de meilleurs résultats, donc à continuer
* insertion de la bathymetry dans un nouveau channel -> ne degrade pas la prediction
* discussion sur la PCA -> donne prediction plus proche de la moyenne, mais n’allege pas vraiment le calcul, contrairement à ce qu’on attendait : parceque dans ce dataset il n’y a pas de tendance, et donc la PCA doit reconstruire du bruit !
* on commence à discuter du banc d’experiences à réaliser pour ajuster la séquence interactive entre simulation explicite et emulation


## 1 avril

* tester normalisation par pavés de 3x3 : differences subtiles, montrer le champ de differences ? resultat final un peu amélioré mais similaire
* padding en place le long des frontieres meridiennes et hemisphere nord -> ne change pas l’apprentissage general (et on en voit rien d’irregulier au voisinage des frontieres, ce qui est tres positif !) -> tester que l’apprentissage est le meme quand on deplace la “couture” à un autre indice (300, 200…)
* comparaison entre 1er mois predit vs ground truth vs mois 12 utilisé dans apprentissage -> 1er mois predit tres similaire à mois 12 (ie persistence plutot que prediction), mais pas exactement (en fait 1er mois predit ressemble à mois 12 + moyenne annuelle)
* toujours pas d’irregularites autour des cotes (incroyable !)
* tests sur apprentissage de la difference entre mois 13 et mois 12 plutot que le mois 13 -> ne marche plus du tout ! c’est lié a priori au fait qu’on essaye alors de predire un ordre de derivation supplementaire, dans un champ bruité, donc plus difficile à predire que le champ direct
* tests sur apprentissages successifs -> l’erreur progresse de maniere lineaire, du 1er au 8e mois
* figure loss par batch (pour 1 epoch) : stabilise rapidement -> utile d’ajouter plus d’epochs ?
* introduire PCA avant CNN pour voir si on obtient meilleurs resultats ?
* ajouter bathymetry dans la couche d’entree (normaliser en divisant par max de la carte globale, avant de l’introduire) -> ameliore la prediction ?


## 24 mars

* tester normalisation par pavés de 3x3 (comme pour structure du CNN) ?
* tester prediction du 12e mois pour voir l’erreur, en particulier sur les zones cotieres ?
* essayer d’injecter prediction du 13e mois dans NEMO pour voir si converge bien vers solution explicite ?
* attention de bien distinguer prediction des petites echelles (“bruit”) de l’evolution de l’etat moyen (tendance basse frequence) -> passer à une autre simulation, où la tendance est forte !
* on peut toujours continuer à jouer avec cette simulation sans derive pour essayer de quantifier jusqu’à combien de mois dans le futur on peut émuler
* pertinence de demande A12 sur V100 seulement (+limité en memoire disponible) ? Nathan peut toujours utiliser qq A100 (80GB) sans demande supplémentaire - ensuite si besoin on fera demande au fil de l’eau aupres du directeur de la Maison de la Simulation / directeur de l’idris


## 23 mars

* travailler sur courbe de prediction, en ciblant qq durées (+1 mois, +50 mois, +100 mois…) pour commencer
* revoir padding (en repliquant i=1 à i=362, idem le long de la frontière nord [attention il faut inverser l’ordre des points])
* sortir courbes de loss function pour voir à quelle vitesse le reseau converge 


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