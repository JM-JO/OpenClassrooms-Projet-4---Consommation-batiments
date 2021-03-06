import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import linregress as reg_lin


def scatter_plot(ser_x,
                 ser_y,
                 min_x=None,
                 max_x=None,
                 min_y=None,
                 max_y=None,
                 alpha=1.0,
                 taille=2,
                 unite_x=None,
                 unite_y=None,
                 droite_identite=False,
                 regression_lineaire=True):
    """ Retourne un scatter plot pour deux séries
    Args :
    - ser_x : série en abcisse.
    - ser_y : série en ordonnée.
    - min_x : valeur min représentée en abcisse.
    - max_x : valeur max représentée en abcisse.
    - min_y : valeur min représentée en ordonnée.
    - max_y : valeur max représentée en ordonnée.
    - alpha : transparence de chaque point. 
    - taille : taille du chaque tracé
    - unite_x : string indiquant l'unité de la série en abcisse.  
    - unite_y : string indiquant l'unité de la série en ordonnée.     
    - droite_identite : booléen pour tracer la droite y=x
    - regression_lineaire : booléen pour tracer la droite de régression linéaire
    Returns : 
    - scatter plot + sa régression linéaire
    """
    
    # Calcul des arguments min_x, max_x, min_y, max_y s'ils sont None :
    if min_x is None:
        min_x = ser_x.min()
    if max_x is None:
        max_x = ser_x.max()
    if min_y is None:
        min_y = ser_y.min()
    if max_y is None:
        max_y = ser_y.max()

    # filtration des valeurs extrêmes. On exclue les valeurs à l'extérieur de l'intervalle [min_x, max_x], considérées comme aberrantes.
    largeur_x = max_x - min_x
    largeur_y = max_y - min_y

    def f_bool(x, min, max):
        if np.isnan(x):
            return False
        elif x < min:
            return False
        elif x > max:
            return False
        else:
            return True

    ser_x_bool = ser_x.apply(lambda x: f_bool(x, min_x, max_x)
                             )  # vaut True pour les valeurs non extrêmes
    ser_y_bool = ser_y.apply(lambda y: f_bool(y, min_y, max_y))
    ser_xy_bool = ser_x_bool & ser_y_bool  # vaut True pour les valeurs non extrêmes de ser_x et de ser_y, sinon vaut False
    # ser_x et ser_y sont les séries tracées
    ser_x = ser_x[ser_xy_bool]
    ser_y = ser_y[ser_xy_bool]
    total_count = ser_xy_bool.sum()

    # mise en place du plot
    plt.style.use('seaborn')
    plt.figure(edgecolor='black', linewidth=4)
    if droite_identite:
        plt.axline([0, 0], [1, 1], c="white", linewidth=1)

    # plotting scatter plot
    plt.scatter(
        ser_x, ser_y, s=taille,
        alpha=alpha)  # la valeur mini de alpha autorisée par pyplot est 0.002
    plt.xlabel(ser_x.name)
    plt.ylabel(ser_y.name)
    plt.title(label='Diagramme de dispersion (Effectif total = ' +
              str(int(total_count)) + ')')
    plt.xlim(min_x - largeur_x / 20, max_x + largeur_x / 20)
    plt.ylim(min_y - largeur_y / 20, max_y + largeur_y / 20)
    label_x = ser_x.name if unite_x is None else ser_x.name + f" ({unite_x})"
    plt.xlabel(label_x)
    label_y = ser_y.name if unite_y is None else ser_y.name + f" ({unite_y})"
    plt.ylabel(label_y)

    # régression linéaire
    if regression_lineaire :
        slope, intercept, r, p, std_err = reg_lin(ser_x, ser_y)
        plt.axline((min_x, min_x * slope + intercept),
                   (max_x, max_x * slope + intercept),
                   ls="--",
                   c="red",
                   linewidth=0.6)
        plt.annotate(
            text="Régression linéaire : y = {0:0.4f}*x + {1:0.4f}, R²={2:0.3f}".
            format(slope, intercept, r * r),
            xy=(min_x + largeur_x / 20, max_y - largeur_y / 20),
            color='red',
            size=10)

    plt.show()
    
    
def display_circles(pca,
                    axis_ranks,
                    labels=None,
                    label_rotation=0,
                    lims=None):
    """ Affiche les cercles des corrélation pour les plan factoriels
    On utilise le code de Nicolas Rangeon (avec qques modifications) disponible ici :
    https://openclassrooms.com/fr/courses/4525281-realisez-une-analyse-exploratoire-de-donnees/5345201-tp-realisez-une-acp
    Args :
    - pca : sklearn.decomposition.PCA
    - axis_ranks : liste des plans factoriel à tracer. Exemple :  [(0,1), (2,3)] pour tracer les deux premiers plans.
    - labels : liste de nom des variables.
    - label_rotation : rotation (degrés) de l'affichage des labels.
    - lims : 'auto' ou 'None' ou tuple (xmin, xmax, ymin, ymax) des limites du tracé.
    Returns : 
    - graphiques matplotlib
    """
    n_comp = pca.n_components_
    for d1, d2 in axis_ranks:  
        if d2 < n_comp:

            # initialisation de la figure
            plt.style.use('seaborn')
            plt.figure(edgecolor='black', linewidth=4, figsize=(10, 10))

            # détermination des limites du graphique
            pcs = pca.components_
            if lims == 'auto':
                xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(pcs[d1, :]), min(
                    pcs[d2, :]), max(pcs[d2, :])
            elif lims is not None:  # lims est un tuple
                xmin, xmax, ymin, ymax = lims
            else:
                xmin, xmax, ymin, ymax = -1, 1, -1, 1

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30:
                plt.quiver(np.zeros(pcs.shape[1]),
                           np.zeros(pcs.shape[1]),
                           pcs[d1, :],
                           pcs[d2, :],
                           angles='xy',
                           scale_units='xy',
                           scale=1,
                           color="grey",
                           width=0.001)

            # affichage des noms des variables
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(x,
                                 y,
                                 labels[i],
                                 fontsize='7',
                                 ha='center',
                                 va='center',
                                 rotation=label_rotation,
                                 color="blue",
                                 alpha=0.5)

            # affichage du cercle
            circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(
                d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(
                d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(
                d1 + 1, d2 + 1))
            plt.axis('square')

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.show()


def display_scree_plot(pca):
    """ Affiche le graphique des éboulis des valeurs propres et le critère de Kaiser
    On utilise le code de Nicolas Rangeon (avec qques modifications) disponible ici :
    https://openclassrooms.com/fr/courses/4525281-realisez-une-analyse-exploratoire-de-donnees/5345201-tp-realisez-une-acp
    Args :
    - pca : sklearn.decomposition.PCA
    Returns : 
    - graphique matplotlib
    """
    scree = pca.explained_variance_ratio_ * 100
    plt.style.use('seaborn')
    plt.figure(edgecolor='black', linewidth=4)
    plt.bar(np.arange(len(scree)) + 1, scree)
    plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red", marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.axhline(100 / pca.n_features_,
                0,
                len(pca.components_) + 1,
                c='g',
                linewidth=0.3)  # critère de Kaiser
    plt.show(block=False)
    
    
def arrondi(nb, chiffres=3, puissance10=None):
    """ Arrondit un nombre et gère les chiffres significatifs.
    Args :
    - nb : nombre (ou un itérable contenant des nombre) à arrondir.
    - chiffres : chiffres significatifs voulus.
    - puissance10 : x est exprimé en puissance 10 de cet argument.
    Returns :
    - le nombre arrondi.
    Examples:
    - arrondi(123.456, chiffres=2) retourne 120.
    - arrondi(0.0123456, chiffres=4) retourne 0.01234.
    - arrondi(123.456, puissance10=2) retourne le string 1.2e2.
    """
    if hasattr(nb, '__iter__'):  
        iterable = True
    else:
        iterable = False
        nb = [nb]     # si nb n'est pas itérable, on le transforme en itérable
    resultats = []
    for x in nb:
        if type(x) is not float:
            x = float(x)
        if x != 0:
            y = round(x, -int(np.floor(np.log10(abs(x)))-chiffres+1))
        else:
            y = 0
        if y >= np.power(10, chiffres-1):
            y = int(y) # on supprime le .0 de tous les floats en Python
        if puissance10 is not None:
            y = "{0:.2f}".format(x/np.power(10, puissance10)) + "e" + str(puissance10)
        resultats.append(y)
    if not iterable :
        resultats = resultats[0]
    return resultats


def display_scores(scores, chiffres=3, puissance10=None):
    """ Transforme la liste de scores retournés par cross_val_score() en valeurs simples à lire.
    Args :
    - scores : valeur fournie par cross_val_score().
    - chiffres : le nombre de chiffres significatifs affichés.
    Returns : 
    - tous les scores de la validation croisée, leur moyenne, leur écart-type
    """
    print("Scores: ", end='')
    for x in scores:
        print(arrondi(x, chiffres, puissance10), end=' ')
    print("\nMean:", arrondi(scores.mean(), chiffres, puissance10))
    print("Standard deviation:", arrondi(scores.std(ddof=1), chiffres, puissance10))
    
    
def barh_plot_coefs(x_array, effectif_array):
    """Retourne un horizontal bar plot pour une série pandas.
    Args :
    - x_array : série avec les longueurs des barres tracées.
    - effectif_array : série avec les effectifs.
    Returns : 
    - horizontal bar plot.
    """
    # Tracé histogramme
    y_array = range(len(x_array))
    labels = x_array.index
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(30, 20), edgecolor='black', linewidth=4)
    ax = fig.add_subplot(111)
    plt.title('Coefficients des différents types de surface (et leur effectif)', size=30)
    plt.xlabel('Coefficient', size=25)
    ax.barh(y_array, x_array, height=0.8, color='lightblue')
    plt.axis(ymin=-0.5, ymax=len(x_array)-0.5)
    plt.xticks(size=18)
    ax.xaxis.get_offset_text().set_size(18)  # taille du multiplicateur (1e7) de coefficients le cas échéants
    plt.yticks([ ])
    plt.tick_params(bottom=True, top=True, direction='in')
    # Annotations
    zip_arrays = zip(y_array, x_array, effectif_array, labels)
    for y, x, effectif, label in zip_arrays:
        plt.annotate(text='  ('+str(effectif)+')  ', xy=(x,y-0.3), size=19, ha='left' if x>0 else 'right')
        plt.annotate(text='   '+str(label)+'   ', xy=(0,y-0.3), size=19, ha='right' if x>0 else 'left')
    plt.show()
    
    
def affichage_tri_r2(cvres) : 
    """Retourne les R² triés pour les différentes valeurs de GridSearchCV
    Args :
    - cvres : paramètre cv_results_ de l'objet grid_search après cross-validation.
    Returns : 
    - R² et les paramètres de grid_search correspondant.
    """
    liste_ordonnee = [x for x in range(len(cvres["mean_test_score"]))]    # on est obligés de rajouter une valeur pour que la méthode sort() ne plante pas en cas d'égalité de score car elle va chercher à comparer des dictionnaires (cvres["params"])
    resultats_zip = zip(cvres["mean_test_score"], liste_ordonnee, cvres["params"])
    resultats_zip = list(resultats_zip)
    resultats_zip
    resultats_zip.sort(reverse=True)   # tri par score
    for mean_score, _, params in resultats_zip:
        print("R² =", str(arrondi(mean_score)).rjust(8, ' '), params)