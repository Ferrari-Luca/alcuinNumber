from project import*

# Exemple de graphe (Le graphe G est un graphe "chevre-loup-salade")
def create_example_graph():
    """
    Crée un graphe où :
    - 1 est la chèvre
    - 2 est le loup
    - 3 est la salade
    Les conflits sont : chèvre-loup et chèvre-salade
    """
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])  # 1: Chèvre, 2: Loup, 3: Salade
    G.add_edges_from([(1, 2), (1, 3)])  # Conflits
    return G


def main():
    # Création du graphe sans le berger
    G = create_example_graph()

    # Capacité du bateau (nombre maximum d'entités transportables)
    k = 1

    # Calcul de la solution
    solution = gen_solution(G, k)

    # Affichage du résultat
    if solution is not None:
        print("Solution trouvée :")
        for step, (berger_side, S0, S1) in enumerate(solution):
            print(f"Étape {step}: Berger sur rive {berger_side}, Rive 0: {S0}, Rive 1: {S1}")
    else:
        print("Aucune solution trouvée.")

if __name__ == "__main__":
    main()


