---
title: Transport optimal appliqué à la recommandation d'œuvres
author: \textsc{Ryan Lahfa}
mathtools: true
homework-assignment: true
numbersections: true
lang: fr
toc: true
loc: true
lof: true
margin-top: 0.3in
margin-bottom: 0.7in
margin-left: 1in
bibliography: true
---

# Position du problème 

## Recommandation d'entrées

À partir d'un ensemble de préférences exprimés par des utilisateurs, on voudrait une méthode de prédiction des futures préférences des utilisateurs. On s'attachera à résoudre ce problème grâce à l'apprentissage automatique supervisée.

Précisément, on se donne une base de données représentée par une matrice $M \in \M_{n, m}(\{0, 1\})$ dont le terme général $(m_{i,j})$ indique si l'utilisateur $i$ a aimé l'entrée $j$.

À partir de $M$, on veut apprendre un classifieur, i.e. $f : [[1, n]] \times [[1, m]] \to \{ 0, 1 \}$, capable de prédire $\widehat{m_{i,j}} = f(i, j)$ si celui-ci n'est pas connu, avec la probabilité, ou non, $p_{i,j} = \PR(m_{i,j} = \widehat{m_{i,j}} \mid \theta) \in [0, 1]$ où $\theta$ est une forme d'information partielle sur $M$.

On notera aussi $r_u \in \R^m$ la distribution de l'utilisateur $u \in [[1, n]]$ sur les entrées.

## Jeu de données: Mangaki

Le modèle sera testé sur le jeu de données fournis par le site [@mangaki] qui comporte:

- 2289 utilisateurs ;
- 12479 œuvres issus de l'animation japonaise (animes, mangas) ;
- plus de 350000 notes

# Modèle de comparaison: 20-KNN

## Introduction

Pour $k \geq 1$ (ici $k = 20$ d'où 20-KNN), le modèle des $k$-plus proches voisins consiste en:

- Pour chaque utilisateur $i$, déterminer les $k$ plus proches voisins, qu'on notera $\mathcal{N}(i) \subset [[1, n]]^k$. Par voisin proche, on entend au sens d'une métrique $d : \R^m \times \R^m \to \R_{+}$ opérant sur les distributions d'utilisateurs.
- Pour classifier une nouvelle entrée pour un utilisateur $i$, on demande l'avis de ses voisins qui connaissent la nouvelle entrée (en question) et on prend l'avis majoritaire sur cette nouvelle entrée comme prédiction, i.e.

En notant $\mathcal{N}'(i)_j = \{ j \in \mathcal{N}(i) \mid m_{u,j} \text{ est connu} \}$ et $m$ son cardinal.
\begin{equation*}
        \widehat{m_{i,j}} =
        \left\{
        \begin{aligned}
                1 & \text{ si } \sum_{k \in \mathcal{N}'(i)_j} m_{k,j} \geq \dfrac{m}{2} \\
                0 & \text{ sinon.}
        \end{aligned}
        \right.
\end{equation*}

On abrégera par KNN le modèle des $k$-plus proches voisins dans le reste du document et 20-KNN pour $k = 20$.

## Choix du paramètre $k$, de la métrique et visualisation des voisins

Le choix du paramètre $k$ peut s'effectuer par validation croisée sur le jeu de données.
Cette validation croisée a été effectuement préalablement et fournit que $k = 20$ donne de bonnes performances relativement à la racine carrée de l'erreur quadratique moyenne (RMSE).

**Remarque 1** : En pratique, on peut apprendre $k$ par recherche d'hyper-paramètres durant l'entraînement du modèle, mais on ne le fera pas en raison d'un coût temporel trop élevé.

Ensuite, pour la métrique, on utilise la similarité cosinus qui possède empiriquement de bonnes performances sur les tâches de recommandation d'après [@sarwar2001item], dont on rappelle la définition:

\begin{equation*}
        \Sim(u, u') = \dfrac{r_u^{\intercal} r_{u'}}{\norm{r_u}_2 \norm{r_{u'}}_2}
\end{equation*}

où $\norm{\cdot}_2$ est la norme $\ell_2$.

\begin{samepage}
        On peut aussi procéder à une visualisation des graphes de voisins dans la figure \ref{knn_graph} et \ref{giant_sub_graph}.
\begin{figure}[H]
        \centering
        \includegraphics[scale=0.5]{assets/knn_graph.png}
        \caption[20 plus proches voisins d'un utilisateur avec plus de 200 notes sur le jeu de données Mangaki]{20 plus proches voisins d'un utilisateur avec plus de 200 notes sur le jeu de données Mangaki\footnotemark} \label{knn_graph}
\end{figure}
\end{samepage}
\begin{figure}[h]
        \centering
        \includegraphics[scale=0.7]{assets/giant_knn_graph.png}
        \caption{Sous-graphe des 20 plus proches voisins sur le jeu de données Mangaki entier} \label{giant_sub_graph}
\end{figure}


## Défauts et limites du modèle

\footnotetext{Les étiquettes sont les identifiants d'utilisateurs}
Comme définit en introduction, l'entraînement de 20-KNN dépend de la métrique $d$ employée. Si on note $\Supp(u)$ pour $u \in [[1, n]]$ le support des utilisateurs, défini par:

\begin{equation*}
        \Supp(u) = \{ j \in [[1, m]] \mid m_{u,j} \text{ est connu} \} 
\end{equation*}

Ainsi, pour $u, v$ deux utilisateurs tels que $\Supp(u) \cap \Supp(v) = \emptyset$, alors: $\Sim(u, v) = 0$ 

Or, la situation dans laquelle l'utilisateur $u$ a lu les versions mangas\footnote{le format livre de l'œuvre} d'une œuvre et $v$ a vu les versions animes\footnote{l'adaptation animée de l'œuvre} de celle-ci peut se présenter, cependant la métrique n'en tient pas compte et ne peut le calculer puisqu'il s'agit d'une information propre à l'œuvre.

## Objectifs du TIPE

Nous répondrons aux questions suivantes:

- Sachant qu'on dispose de l'ensemble des couvertures des œuvres, peut-on calculer une métrique qui tient compte de l'information visuelle de ces couvertures et des similarités entre les distributions d'utilisateurs ?
- En la remplaçant par la similarité cosinus, obtient-on un meilleure performance au sens d'une métrique d'erreur ?
- Est-ce qu'on constate des transferts d'information\footnote{Cela se manifeste par l'existence d'une dépendance entre la variable aléatoire qui indique si un} pertinents et intéressants tels que: la saison $i$ d'une œuvre vers la saison $i + j$ de la même œuvre, du format manga vers le format anime ou vice versa ?

Ces travaux sont motivés notamment par [@vie2017] et forme un prolongement possible de cet article.

## État actuel de la recherche

À notre connaissance, la littérature ne mentionne pas beaucoup de travaux qui cherche à intégrer des métadonnées visuelles dans un système de recommandation afin d'en améliorer sa qualité et son interprétabilité.

On notera: [@chu2017hybrid], [@prompt] et

- [@vie2017] où il s'agit d'un modèle qui combine un filtrage collaboratif et ensemble de régresseurs linéaires par utilisateur afin d'apprendre des préférences visuelles dans un cadre de démarrage à froid, ce travail permet l'interprétabilité des goûts d'un utilisateur en inspectant la matrice du régresseur linéaire, travaux qui ont inspiré celui-ci ;
- [@messina2019] où il s'agit d'un modèle purement basé sur le contenu qui extrait automatiquement des représentations visuelles profondes et des métadonnées telles que le contraste afin de recommander des œuvres artistiques digitales-- « artworks ».

# Raffinement par le transport optimal: impact de la distance de Wasserstein

Le transport optimal est un domaine qui est de plus en plus appliqué notamment grâce à [@cuturi2013] qui a permis le calcul effectif et approximatif des objets de façon tractable.

Au préalable, notons $\Sigma_d = \{ x \in \R_{+}^d \mid \sum_{i=1}^d x_i = 1 \}$ le simplexe de dimension $d$, qui peut s'interpréter de façon probabiliste comme une distribution de probabilité discrète à valeurs dans $[[1, d]]$.

Si l'on dispose de $r, c \in \Sigma_d$ deux distributions de probabilités discrètes.

On pose $U(r, c) = \{ M \in \M_{d, d}(\R_{+}) \mid M \mathbbm{1}_d = r \text{ et } M^{\intercal} \mathbbm{1}_d = c \}$, l'ensemble des probabilités jointes sur $r$ et $c$ à valeurs dans $[[1, d]]^2$.

Ainsi, on définit la distance de Wasserstein comme étant:

\begin{equation*}
        \mathcal{W}(r, c) = \min_{\gamma \in U(r, c)} \dps{\gamma}{C}_F
\end{equation*}

où $C$ est une matrice exprimant le coût de transporter de la masse de $r_i$ vers $c_j$ et $\dps{\cdot}{\cdot}_F$ est le produit scalaire de Frobenius.

**Remarque** : On rencontre aussi le nom de « Earth's Mover Distance » ou EMD pour la distance de Wasserstein, cela peut s'expliquer par l'interprétation intuitive suivante: si les distributions $r, c$ sont des masses, au sens physique, renormalisés à 1, et que la matrice $C$ représente des distances euclidiennes physiques de transporter les masses de $r$ vers les masses de $c$, alors la distance de Wasserstein est le coût minimal de transport afin de transformer une masse en l'autre par déplacements successifs.

## Autour de $\mathcal{W}$

$\mathcal{W}$ est bien une distance sur les distributions de probabilités (discrètes), démontrée dans [@villani2008], ce qui motive son usage en tant que métrique pour KNN.

En revanche, son temps de calcul est prohibitif, en effet, il est en $O(d^3 \log d)$ au mieux, par des variations de l'algorithme du simplexe sur un graphe, que l'on ne détaillera pas puisque cela dépasse largement le cadre de ce TIPE, d'après [@pele2009fast].

## Intérêt: calcul efficace et rapide $\mathcal{W}$, propagation de l'information visuelle dans le modèle

On introduit $\varepsilon > 0$, un paramètre de régularisation entropique, qui se justifie pour deux raisons:

- Rendre le calcul plus rapide ;
- Rendre le plan de transport optimal le plus simple\footnote{lire: parcimonieux, avec le plus de zéros} possible en forçant son entropie à être faible

\begin{equation*}
        \mathcal{W}_{\varepsilon}(r, c) = \min_{\gamma \in U(r, c)} \dps{\gamma}{C}_F + \varepsilon \Omega(\gamma)
\end{equation*}

De plus, le minimum est atteint pour un $\gamma$ unique et la solution est de la forme:

\begin{equation*}
        \forall (i, j) \in [[1, d]]^2, \gamma_{i,j} = u_i \exp(-C/\varepsilon) v_j
\end{equation*}

Pour $(\avec{u}, \avec{v}) \in (\R^d)^2$ des facteurs de dilations qu'on calcule itérativement.

C'est l'algorithme de Sinkhorn-Knopp, présenté initialement dans [@sinkhorn1967diagonal], réutilisé de façon fondamentale dans [@cuturi2013]. Il permet donc de calculer une approximation de $\mathcal{W}$ en temps quadratique en la dimension $d$.

On prouve aussi dans [@cuturi2013] que $(r, c) \mapsto \mathbbm{1}_{r \neq c} \mathcal{W}_{\varepsilon}(r, c)$ est une distance par la même approche employée dans [@villani2008].

**Remarque 1** : En raison de la nature itérative de l'algorithme de Sinkhorn-Knopp, on peut très facilement paralléliser le calcul des facteurs de dilatations (qui sont des vecteurs dans l'algorithme original) en des matrices lorsqu'on calcule des distances d'une distribution de probabilité vers $M$ distributions de probabilités.

## Calcul des représentations visuelles par le réseau de neurones convolutifs Illustration2Vec

En utilisant les travaux de [@saito2015] et [@vie2017], on peut calculer des représentations parcimonieuses de couvertures $(p_i)_i \in \left(\R^{512}\right)^m$ qui permettent de poser la matrice de coût comme étant:

\begin{equation*}
        C = \left(\norm{p_i - p_j}_2^2\right)_{(i, j) \in [[1, m]]^2}
\end{equation*}

\begin{samepage}
Donc, la matrice de coût représente la similarité visuelle entre deux couvertures, que l'on illustre entre les deux saisons de l'anime \textbf{Code Geass: Lelouch of the Rebellion} :

\begin{figure}[!ht]
        \centering
        \includegraphics[scale=0.4]{assets/cg_similarity.png}
        \caption{Distance entre les deux couvertures où le coût de déplacement est minimal sur toutes les œuvres du jeu de données}
\end{figure}
\end{samepage}

En vertu de cette propriété de coût minimal, la distance de Wasserstein propagera correctement l'information visuelle entre les deux couvertures, qui s'avère être l'information : « ces deux œuvres font partis du même univers et sont directement la suite ou l'antépisode l'un de l'autre ».

Le troisième objectif du TIPE **a été donc atteint**.

On appellera donc désormais 20-W-KNN le modèle 20-KNN dans lequel on remplace la similarité cosinus par une approximation de la distance de Wasserstein $\mathcal{W}$ avec la matrice $C$ définie comme précédemment.

Le premier objectif du TIPE **a été atteint**.

## Calcul des prédictions de 20-WKNN

Dans le modèle à similarité cosinus, on fait voter les voisins afin de calculer une prédiction.
En revanche, puisqu'on manipule des distributions de probabilités, on ne peut plus procéder à la détermination de la classe majoritaire.

Au lieu de cela, on calcule le barycentre des distances de Wasserstein, pour un utilisateur $u$ donnée:

\begin{equation*}
        v = \dfrac{1}{\card \mathcal{N}(u)} \argmin_{v \in \Sigma_m} \sum_{u' \in \mathcal{N}(u)} \mathcal{W}(v, u')
\end{equation*}

**Remarque 1** : Ce calcul peut être encore effectué rapidement par l'algorithme de Sinkhorn-Knopp, d'après [@cuturi2014fast].

Ainsi, on connaît le germe de probabilité:

\begin{equation*}
        \left(\PR(m_{u,j} = 1)\right)_{j \in [[1, m]]} = (v_j)_{j \in [[1, m]]}
\end{equation*}

À partir de cela, on peut opter pour une méthode à base de seuil, on fixe un paramètre $\alpha \in [0, 1]$ et on pose:

\begin{equation*}
        \widehat{m_{u,j}} = \left\{
                \begin{aligned}
                        & 1 \text{ si } \PR(m_{u,j} = 1) \geq \alpha \\
                        & 0 \text{ sinon.}
                \end{aligned}
                \right.
\end{equation*}

On appelle $\alpha$ seuil de discrimination.

**Remarque 1** : Ce paramètre peut être appris par recherche d'hyper-paramètres pendant l'entraînement du modèle.

**Remarque 2** : Il peut être rendu dépendant de l'utilisateur, lors de l'entraînement en déterminant le profil d'un utilisateur. Par exemple, une approche naïve consisterait à dire que plus un utilisateur a de préférences négatives, plus le seuil de discrimination sera élevé, et vice versa.

**Remarque 3** : Si les $\alpha$ sont dépendants des utilisateurs, on peut calculer un $\overline{\alpha}$ moyen que l'on peut employer pour une prédiction sur un nouvel utilisateur qui n'était pas présent dans la phase d'entraînement.

# Résultats

Les code des expériences sont fournies sur le référentiel GitHub: <https://github.com/mangaki/hiyajo-ot> et reproducibles.

Le matériel employé pour l'expérience est un serveur muni d'un Intel(R) Atom(TM) CPU C2750  @ 2.40GHz à 8 cœurs et 16 Gio de RAM.

Plusieurs implémentations de référence seront réutilisés directement plutôt que de les réécrire car leur (ré)-implémentation ne concerne pas le fond de ce TIPE, on utilisera notamment NumPy, SciPy, NetworkX, IPyParallel et enfin POT [@flamary2017pot] qui fournit des implémentations de l'algorithme de Sinkhorn.

## Évaluation de l'erreur de recommandation: courbe ROC

Une courbe ROC est un graphique exprimant le taux de vrai positifs (TPR) par rapport au taux de faux positifs (FPR).
On calcule l'ensemble des seuils de discriminations possibles et on trace une courbe ROC montrant ces seuils et on peut calculer l'aire sous la courbe, ce résultat est indépendant des seuils de discrimination.
Ce qui rend le choix de cet outil intéressant puisque 20-WKNN dépend d'un seuil de discrimination pour effectuer des prédictions. Ce résultat est pris comme l'erreur de recommandation que l'on note en tant que « AUROC » pour « Area Under Receiving Operating Characteristics ».

\begin{figure}[!ht]
\centering
\begin{tabular}{cccc} \toprule
AUROC & Ensemble de test\\ \midrule
KNN & 0.514 \\
W-KNN & \textbf{0.625}\\ \bottomrule
\end{tabular}
\caption{$5$-fold où l'on s'assure que les classes sont balancés, répétés 3 fois avec un seed de $42$}
\end{figure}

Le second objectif du TIPE **a été atteint**, on constate que W-KNN a une meilleure performance que KNN de façon stable et reproductible.

# Prolongements envisagables

Après avoir atteint tous les objectifs fixés par le TIPE.
Nous n'avons pas discuté des différences entre les temps d'entraînement et de prédiction de KNN et W-KNN.
En l'état, W-KNN est 100 fois plus lent à entraîner que KNN malgré l'algorithme de Sinkhorn, cela s'explique que les expériences n'ont pas exploité le caractère hautement parallélisable de cet algorithme afin de faire diminuer les temps de calculs.

Cependant, une alternative est envisageable par [@altschuler2017near], un algorithme quasi-linéaire de calcul des distances approximatifs de Wasserstein est possible, et mis en place sous le nom de **Greenkhorn** dans la librairie POT, cependant il est très sensible aux erreurs numériques et n'est pas conçu pour la parallélisation automatique. Durant les essais préliminaires, aucune renormalisation n'a été fructueux.

De plus, l'algorithme de Sinkhorn possède une version stabilisée qui fonctionne selon le principe décrit dans [@schmitzer2019stabilized] mais Greenkhorn n'en possède aucune, il serait intéressant de contribuer à POT afin d'ajouter le support pour un tel schéma de calcul numérique et de résoudre le problème décrit ici: <https://github.com/rflamary/POT/issues/54> par la même occasion.

# Réferences bibliographiques
