from project import*

if __name__ == "__main__":
    # Simple example

    # Define the graph G (e.g., the goat, wolf, and cabbage problem)
    G = nx.Graph()
    G.add_edges_from([
        ('goat', 'wolf'),
        ('goat', 'cabbage')
    ])

    # Call the gen_solution function
    k = 2  # Boat capacity
    solution = gen_solution(G, k)

    if solution is not None:
        for t, (berger_side, S0_t, S1_t) in enumerate(solution):
            print(f"Time {t}:")
            print(f"  Berger is on shore {berger_side}")
            print(f"  Shore 0: {S0_t}")
            print(f"  Shore 1: {S1_t}")
    else:
        print("No solution found.")

    #test nombre d'Alcuin
    G = nx.Graph()
    G.add_edges_from([
        ('chèvre', 'loup'),
        ('chèvre', 'chou')
    ])

    # Calcul du nombre d'Alcuin
    alcuin_number = find_alcuin_number(G)
    print(f"Le nombre d'Alcuin du graphe est : {alcuin_number}")

    # Define the graph G (e.g., the goat, wolf, and cabbage problem)
    G = nx.Graph()
    G.add_edges_from([
        ('goat', 'wolf'),
        ('goat', 'cabbage')
    ])

    # Call the gen_solution_cvalid function
    k = 2  # Boat capacity
    c = 2  # Number of compartments
    solution = gen_solution_cvalid(G, k, c)

    if solution is not None:
        for t, (berger_side, S0_t, S1_t, compartments_t) in enumerate(solution):
            print(f"Time {t}:")
            print(f"  Berger is on shore {berger_side}")
            print(f"  Shore 0: {S0_t}")
            print(f"  Shore 1: {S1_t}")
            if t < len(solution) - 1:
                print(f"  Compartments during move:")
                for idx, compartment in enumerate(compartments_t):
                    print(f"    Compartment {idx + 1}: {compartment}")
    else:
        print("No solution found.")

