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

À partir d'un ensemble de préférences exprimés par des utilisateurs, l'on veut une méthode de prédire les futures préférences des utilisateurs, c'est le problème qu'on essayera de résoudre.

Précisément, on se donne une base de données représentée par une matrice $M \in \M_{n, m}(\{0, 1\})$ dont le terme général $(m_{i,j})$ indique si l'utilisateur $i$ a aimé l'entrée $j$.

À partir de $M$, l'on veut apprendre un classificateur capable de prédire $\widehat{m_{i,j}}$ si celui-ci n'est pas connu, avec, ou non, $p_{i,j} = \PR(m_{i,j} = \widehat{m_{i,j}} \mid \theta) \in [0, 1]$ où $\theta$ est une forme d'information partielle sur $M$.

On notera aussi $r_u \in \R^m$ la distribution de l'utilisateur $u \in [[1, n]]$ sur les entrées.

## Jeu de données: Mangaki

Le modèle sera testé sur le jeu de données fournis par le site [@mangaki] qui comporte:

- 2289 utilisateurs ;
- 12479 œuvres issus de l'animation japonaise (animes, mangas) ;
- plus de 350000 notes

# Modèle de comparaison: 20-KNN

## Introduction

Pour $k \geq 1$ (ici $k = 20$ d'où 20-KNN), le modèle des $k$-plus proches voisins consiste en:

- Pour chaque utilisateur $i$, calculer des voisins, qu'on notera $\mathcal{N}(i) \subset [[1, n]]^k$ au sens d'une métrique $d : \R^m \times \R^m \to \R_{+}$ opérant sur les distributions d'utilisateurs, prendre les $k$ plus proches
- Pour classifier une nouvelle entrée pour un utilisateur $i$, on fait « voter » les voisins de $i$ et on prend la majorité comme prédiction, i.e.

En notant $\mathcal{N}'(i)_j = \{ j \in \mathcal{N}(i) \mid m_{u,j} \text{ est connu} \}$ et $m$ son cardinal.
\begin{equation*}
        \widehat{m_{i,j}} =
        \left\{
        \begin{aligned}
                1 & \text{ si } 2\sum_{k \in \mathcal{N}'(i)_j} m_{k,j} \geq m \\
                0 & \text{ sinon.}
        \end{aligned}
        \right.
\end{equation*}

On abrégera KNN pour le modèle des $k$-plus proches voisins dans le reste du document et 20-KNN pour $k = 20$.

## Choix du paramètre $k$, de la métrique et visualisation des voisins

Le choix du paramètre $k$ peut s'effectuer par validation croisée sur le jeu de données, cette validation croisée a été effectuement préalablement et fournit que $k = 20$ donne une bonne performance relativement à la racine carrée de l'erreur moyenne au carrée (RMSE).

Ensuite, pour la métrique, on utilise la similarité cosinus qui possède de bonnes performances empiriquement sur les tâches de recommandation d'après [@sarwar2001item], dont on rappelle la définition:

\begin{equation*}
        sim(u, u') = \dfrac{r_u^{\intercal} r_{u'}}{\norm{r_u}_2 \norm{r_{u'}}_2}
\end{equation*}

où $\norm{\cdot}_2$ est la norme $\ell_2$.

On peut aussi procéder à une visualisation des graphes de voisins:

## Défauts et limites du modèle

En introduction, l'entraînement de 20-KNN dépend de la métrique $d$ employée, si on note $\Supp(u)$ pour $u \in [[1, n]]$ le support des utilisateurs, défini par:

\begin{equation*}
        \Supp(u) = \{ j \in [[1, m]] \mid m_{u,j} \text{ est connu} \} 
\end{equation*}

Alors, pour $u, v$ deux utilisateurs tels que $\Supp(u) \cap \Supp(v) = \emptyset$, alors: $sim(u, v) = 0$ 

Or, la situation dans laquelle l'utilisateur $u$ a lu les versions mangas d'une œuvre et $v$ a vu les versions animes de celle-ci peut se présenter, cependant la métrique n'en tient pas compte et ne peut le calculer puisqu'il s'agit d'une information propre à l'œuvre.

## Objectifs du TIPE

Nous répondrons aux questions suivantes:

- Sachant qu'on dispose de l'ensemble des couvertures des œuvres, peut t'on calculer une métrique qui tient compte de l'information visuelle de ces couvertures et des similarités entre les distributions d'utilisateurs ?
- En la remplaçant par la similarité cosinus, obtient t'on un meilleure performance au sens d'une métrique d'erreur ?
- Est-ce qu'on constate des transferts d'information pertinents et intéressants tels que: la saison $i$ d'une œuvre vers la saison $i + j$ de la même œuvre, du format manga vers le format anime ou vice versa ?

Ces travaux sont motivés notamment par [@vie2017] et forme un prolongement possible de cet article.

## État actuel de la recherche

À notre connaissance, la littérature ne mentionne pas beaucoup de travaux qui chercher à intégrer des métadonnées visuelles dans un système de recommandation afin d'en améliorer sa qualité et son interprétabilité, on notera [@vie2017] où il s'agit d'un modèle qui combine un ensemble de régresseurs linéaires par utilisateur afin d'apprendre des préférences visuelles dans un cadre de démarrage à froid, ce travail permet l'interprétabilité des goûts d'un utilisateur en inspectant la matrice du régresseur linéaire, [@messina2019]

# Raffinement par le transport optimal: impact de la distance de Wasserstein

Le transport optimal est un domaine qui est de plus en plus appliqué notamment grâce à [@cuturi2013] qui a permis le calcul effectif et approximatif des objets de façon tractable.

Si l'on dispose de $r \in \R^d, c \in \R^d$ deux distributions de probabilités discrètes, en posant $U(r, c) = \{ M \in \M_{d, d}(\R_{+}) \mid M \mathbbm{1}_d = r \text{ et } M^{\intercal} \mathbbm{1}_d = c \}$, l'ensemble des probabilités jointes sur $r$ et $c$, on définit la distance de Wasserstein comme étant:

\begin{equation*}
        \mathcal{W}(r, c) = \min_{\gamma \in U(r, c)} \dps{\gamma}{C}_F
\end{equation*}

où $C$ est une matrice exprimant le coût de transporter de la masse de $r_i$ vers $c_j$ et $\dps{\cdot}{\cdot}_F$ est le produit scalaire de Frobenius.

## Propriétés de $\mathcal{W}$

La distance de Wasserstein est une métrique.

## Intérêt: calcul efficace et rapide $\mathcal{W}$, propagation de l'information visuelle dans le modèle

Par l'algorithme de Sinkhorn, présenté en détails dans [@cuturi2013], il est possible de calculer une approximation de $\mathcal{W}$, pour $\varepsilon > 0$, un paramètre de régularisation entropique :

\begin{equation*}
        \mathcal{W}_{\varepsilon}(r, c) = \min_{\gamma \in U(r, c)} \dps{\gamma}{C}_F + \varepsilon \Omega(\gamma)
\end{equation*}

On prouve aussi dans [@cuturi2013] que $(r, c) \mapsto \mathbbm{1}_{r \neq c} \mathcal{W}_{\varepsilon}(r, c)$ est une distance.

## Calcul des représentations visuelles par le réseau de neurones convolutifs Illustration2Vec

En utilisant les travaux de [@saito2015] et [@vie2017], on peut calculer des représentations parcimonieuses de couvertures $(p_i)_i \in \left(\R^{512}\right)^m$ qui permettent de poser la matrice de coût comme étant:

\begin{equation*}
        C = \left(\norm{p_i - p_j}_2^2\right)_{(i, j) \in [[1, m]]^2}
\end{equation*}

Donc, la matrice de coût représente la similarité visuelle entre deux couvertures.

# Résultats

## AUC

## Temps de calcul

## Analyse qualitative

# Prolongements envisagables

# Réferences bibliographiques
