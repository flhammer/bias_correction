# Tous les scripts du dernier downscaling en date, à adapter pour d'autres domaines / modèles

Pour Nora Helbig

En adaptant les scripts ci-dessous il devrait être possible de faire des downscaling de vent sur vos domaines. Quelques commentaires pour vous aider dans cette tâche.



- partir de `4_compute_corrections_again.py`
	- unet_path (à modifier): c'est le dossier contenant les poids du réseau, je l'ai mis dans le dépot git dans `coeffs_unet/date_21_12_2021_name_simu_classic_all_low_epochs_0_model_UNet/`
	- c'est un wrapper autour de la fonction de Louis `bias_correction.train.model.CustomModel.predict_multiple_batches`
	- pour que ça aille vite il faut le faire tourner sur un serveur avec GPU (ça se fait aussi sans GPU mais c'est longuet...)
	- selon la parité des dimensions du domaine le code peut planter. Je ne sais pas exactement quelles sont les conditions précises pour que ça tourne mais il faut faire des essais, rogner les deux dimensions du domaine de 1, 2, 3 pixels jusqu'à ce que ça marche
	- j'ai utilisé docker pour faire tourner la partie GPU ce n'est pas indispensable mais c'est pratique (si vous avez un environnement python avec  tensorflow et  GPU qui fonctionne vous pouvez l'utiliser) - je donne tout de même quelques détails car le code fait référence à cet environnement (je vous laisse adapter le code)
		- le `dockerfile` à la racine du projet m'a servi à construire le container docker sur le serveur sur lequel je fais tourner le code, le container est lancé avec la commande:  `docker run -u $(id -u):$(id -g) -it --rm -v $HOME/bias_correction:/app/code -v $HOME/bias_correction_data:/app/data devine bash -c "pip install -e . && bash"`
		- `-v $HOME/bias_correction:/app/code -v $HOME/bias_correction_data:/app/data` fait le mapping entre les paths sur le serveur  physique et les path à l'intérieur du container (dans le code on fait référence aux paths du container docker)
		- les dépendances ont été installées à la fabrication de l'image docker mais l'installation du projet se fait au lancement (`pip install -e .`) cela permet que les modifications du code soient prises en compte immédiatement dans le container 

- `6_prepare_data.py` c'est pour lire les données modèles dans nos bases en utilisant un code que j'ai développé par ailleurs celà ne vous intéresse pas.
- `7_arome_downscaling.py`
	- interpolation d'arome à la résolution visée (en utilisant une triangulation de Delaunay) puis application des correction de Devine aux champs de vent interpolés
	- si on traite des grandes séries temporelles, le code est assez rapide par contre il consomme une grande quantité de mémoire vive il aurait du mal à tourner sur un PC perso!
	- on a besoin d'un dataset de vent (arome dans le code) qui a des variables `u` et `v` (deux composantes du vent), dans le code on le passe en epsg:2154 car c'est le système de coordonnées du DEM
	- note j'ai réécrit la fonction d'interpolation (`get_interpolated_wind`) afin de calculer les coefficients d'interpolation une seule fois et les appliquer à chaque pas de temps (ils ne dépendent pas du temps)

