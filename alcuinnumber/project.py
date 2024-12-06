import networkx as nx

# Q2
from pysat.solvers import Minicard
from pysat.formula import CNFPlus, IDPool, CNF

import networkx as nx
from itertools import combinations



def gen_solution(G: nx.Graph, k: int) -> list[tuple[int, set, set]]:
    """
    Génère une solution pour le graphe G avec une capacité de bateau k,
    en traduisant les contraintes de cardinalité directement en clauses FNC.

    Paramètres:
    - G: Graphe non orienté (networkx.Graph)
    - k: Capacité maximale du bateau (entier)

    Retourne:
    - Une liste de triplets (berger_side, S0_t, S1_t) pour chaque instant t,
      ou None si aucune solution n'est trouvée.
    """
    # Initialize the variable ID pool
    vpool = IDPool()

    subjects = list(G.nodes())
    E = list(G.edges())
    n = len(subjects)

    # Estimate the maximum number of steps (T) needed
    # We can use T = 2n + 1 as per Theorem 1
    T = 2 * n + 1

    # Initialize the CNF formula
    cnf = CNF()

    # Variables:
    # L_s_t: Subject s is on shore 0 at time t
    # B_t: Berger is on shore 0 at time t
    # M_s_t: Subject s moves at time t

    # Create variables
    L = {}  # L[s][t]
    B = {}  # B[t]
    M = {}  # M[s][t]

    for s in subjects:
        L[s] = {}
        M[s] = {}
        for t in range(T + 1):
            L[s][t] = vpool.id(f'L_{s}_{t}')
            if t < T:
                M[s][t] = vpool.id(f'M_{s}_{t}')

    for t in range(T + 1):
        B[t] = vpool.id(f'B_{t}')

    # Initial Conditions
    # All subjects and berger start on shore 0 at time 0
    for s in subjects:
        cnf.append([L[s][0]])  # L_s_0 is True
    cnf.append([B[0]])  # B_0 is True

    # Final Conditions
    # All subjects and berger end on shore 1 at time T
    for s in subjects:
        cnf.append([-L[s][T]])  # L_s_T is False
    cnf.append([-B[T]])  # B_T is False

    # Berger's movement: Berger alternates shores
    for t in range(T):
        # B_{t+1} <-> ~B_t
        cnf.extend([
            [-B[t], -B[t+1]],
            [B[t], B[t+1]]
        ])

    # Movement Variables and Constraints
    for s in subjects:
        for t in range(T):
            # M_{s,t} <-> (L_{s,t} XOR L_{s,t+1})
            # Encoding XOR:
            cnf.extend([
                [-M[s][t], L[s][t], L[s][t+1]],
                [-M[s][t], -L[s][t], -L[s][t+1]],
                [M[s][t], -L[s][t], L[s][t+1]],
                [M[s][t], L[s][t], -L[s][t+1]]
            ])

            # Subjects move with the berger
            # M_{s,t} -> (L_{s,t} == B_t)
            cnf.extend([
                [-M[s][t], -L[s][t], B[t]],
                [-M[s][t], L[s][t], -B[t]]
            ])

    # Boat capacity constraints
    # At each time t, sum of M_{s,t} <= k
    for t in range(T):
        m_vars = [M[s][t] for s in subjects]
        if len(m_vars) > k:
            # Generate all combinations of k+1 variables
            for combo in combinations(m_vars, k + 1):
                # Add clause: At least one of these variables is False
                cnf.append([-var for var in combo])

    # Conflict constraints
    # At each time t, no conflicting subjects are left alone without the berger
    for t in range(T + 1):
        for (i, j) in E:
            # (~L_{i,t} OR ~L_{j,t} OR B_t)
            cnf.append([
                -L[i][t], -L[j][t], B[t]
            ])
            # (L_{i,t} OR L_{j,t} OR ~B_t)
            cnf.append([
                L[i][t], L[j][t], -B[t]
            ])

    # Solve the CNF formula
    solver = Minicard(bootstrap_with=cnf.clauses)
    if solver.solve():
        model = solver.get_model()
        solver.delete()

        # Extract the solution
        result = []
        for t in range(T + 1):
            # Accéder directement au côté du berger
            berger_side = 0 if model[B[t] - 1] > 0 else 1

            # Extraire les sujets sur chaque rive
            S0_t = {s for s in subjects if model[L[s][t] - 1] > 0}
            S1_t = set(subjects) - S0_t

            # Ajouter l'étape à la solution
            result.append((berger_side, S0_t, S1_t))
        return result
    else:
        solver.delete()
        return None



# Q3
def find_alcuin_number(G: nx.Graph) -> int:
    """
    Calcule le nombre d'Alcuin du graphe G en utilisant la fonction gen_solution
    avec une recherche dichotomique

    Paramètres:
    - G: un graphe non orienté (objet networkx.Graph)

    Retourne:
    - Le nombre d'Alcuin de G (un entier)
    """
    n = len(G.nodes)
    left = 1
    right = n
    alcuin_number = None

    while left <= right:
        mid = (left + right) // 2
        solution = gen_solution(G, mid)
        if solution is not None:
            alcuin_number = mid
            right = mid - 1
        else:
            left = mid + 1
    return alcuin_number



# Q5
def gen_solution_cvalid(G: nx.Graph, k: int, c: int) -> list[tuple[int, set, set, tuple[set]]]:
    """
    Generate a c-valid solution for the given graph G with boat capacity k and c compartments.

    Parameters:
    - G: A networkx.Graph representing the problem graph.
    - k: An integer representing the maximum number of subjects transported simultaneously.
    - c: An integer representing the number of compartments in the boat.

    Returns:
    - A list of tuples (berger_side, S0_t, S1_t, compartments_t) for each time step, where:
      - berger_side: 0 or 1 indicating the berger's position.
      - S0_t: Set of subjects on shore 0 at time t.
      - S1_t: Set of subjects on shore 1 at time t.
      - compartments_t: Tuple of sets, each set contains subjects in a compartment during the move.
    - Returns None if no solution is found.
    """
    # Initialize the variable ID pool
    vpool = IDPool()

    subjects = list(G.nodes())
    E = list(G.edges())
    n = len(subjects)

    # Estimate the maximum number of steps (T) needed
    # We can use T = 2n + 1 as per Theorem 1
    T = 2 * n + 1

    # Initialize the CNF formula
    cnf = CNF()

    # Variables:
    # L_s_t: Subject s is on shore 0 at time t
    # B_t: Berger is on shore 0 at time t
    # M_s_t: Subject s moves at time t
    # C_s_t_d: Subject s is assigned to compartment d at time t

    # Create variables
    L = {}  # L[s][t]
    B = {}  # B[t]
    M = {}  # M[s][t]
    C = {}  # C[s][t][d]

    for s in subjects:
        L[s] = {}
        M[s] = {}
        C[s] = {}
        for t in range(T + 1):
            L[s][t] = vpool.id(f'L_{s}_{t}')
            if t < T:
                M[s][t] = vpool.id(f'M_{s}_{t}')
                C[s][t] = {}
                for d in range(1, c + 1):
                    C[s][t][d] = vpool.id(f'C_{s}_{t}_{d}')

    for t in range(T + 1):
        B[t] = vpool.id(f'B_{t}')

    # Initial Conditions
    # All subjects and berger start on shore 0 at time 0
    for s in subjects:
        cnf.append([L[s][0]])  # L_s_0 is True
    cnf.append([B[0]])  # B_0 is True

    # Final Conditions
    # All subjects and berger end on shore 1 at time T
    for s in subjects:
        cnf.append([-L[s][T]])  # L_s_T is False
    cnf.append([-B[T]])  # B_T is False

    # Berger's movement: Berger alternates shores
    for t in range(T):
        # B_{t+1} <-> ~B_t
        cnf.extend([
            [-B[t], -B[t+1]],
            [B[t], B[t+1]]
        ])

    # Movement Variables and Constraints
    for s in subjects:
        for t in range(T):
            # M_{s,t} <-> (L_{s,t} XOR L_{s,t+1})
            # Encode XOR as:
            cnf.extend([
                [-M[s][t], L[s][t], L[s][t+1]],
                [-M[s][t], -L[s][t], -L[s][t+1]],
                [M[s][t], -L[s][t], L[s][t+1]],
                [M[s][t], L[s][t], -L[s][t+1]]
            ])

            # Subjects move with the berger
            # M_{s,t} -> (L_{s,t} == B_t)
            cnf.extend([
                [-M[s][t], -L[s][t], B[t]],
                [-M[s][t], L[s][t], -B[t]]
            ])

            # Compartment Assignment Constraints
            # If M_{s,t} is true, then s must be assigned to exactly one compartment
            # M_{s,t} -> (C_{s,t,1} or ... or C_{s,t,c})
            cnf.append([-M[s][t]] + [C[s][t][d] for d in range(1, c + 1)])

            # For uniqueness: s cannot be assigned to more than one compartment
            for d1 in range(1, c + 1):
                for d2 in range(d1 + 1, c + 1):
                    cnf.append([-C[s][t][d1], -C[s][t][d2]])

            # If M_{s,t} is false, then C_{s,t,d} must be false for all d
            for d in range(1, c + 1):
                cnf.append([M[s][t], -C[s][t][d]])

    # Boat capacity constraints
    # At each time t, sum of M_{s,t} <= k
    for t in range(T):
        m_vars = [M[s][t] for s in subjects]
        if len(m_vars) > k:
            # Generate all combinations of k+1 variables
            for combo in combinations(m_vars, k + 1):
                # Add clause: At least one of these variables is False
                cnf.append([-var for var in combo])

    # Conflict constraints
    # At each time t, no conflicting subjects are left alone on a shore without the berger
    for t in range(T + 1):
        for (i, j) in E:
            # (~L_{i,t} OR ~L_{j,t} OR B_t)
            cnf.append([
                -L[i][t], -L[j][t], B[t]
            ])
            # (L_{i,t} OR L_{j,t} OR ~B_t)
            cnf.append([
                L[i][t], L[j][t], -B[t]
            ])

    # No conflicts within compartments during movement
    # For all t, for all compartments d, no two conflicting subjects are assigned to the same compartment if they are moving
    for t in range(T):
        for (i, j) in E:
            for d in range(1, c + 1):
                cnf.append([
                    -M[i][t], -M[j][t], -C[i][t][d], -C[j][t][d]
                ])

    # Solve the CNF formula
    solver = Minicard(bootstrap_with=cnf.clauses)
    if solver.solve():
        model = solver.get_model()
        solver.delete()

        # Extract the solution
        result = []
        for t in range(T + 1):
            berger_on_shore_0 = (B[t] in model)

            S0_t = set()
            S1_t = set()
            for s in subjects:
                if L[s][t] in model:
                    S0_t.add(s)
                else:
                    S1_t.add(s)

            berger_side = 0 if berger_on_shore_0 else 1

            # For t in [0, T), extract compartments during movement
            if t > 0:
                compartments_t = [set() for _ in range(c)]
                for s in subjects:
                    # Les déplacements qui ont mené à l'instant t sont encodés à l'instant t-1
                    if M[s][t - 1] in model:
                        # Le sujet s s'est déplacé entre t-1 et t
                        for d in range(1, c + 1):
                            if C[s][t - 1][d] in model:
                                compartments_t[d - 1].add(s)
                                break
                compartments = tuple(compartments_t)
            else:
                compartments = tuple()

            result.append((berger_side, S0_t, S1_t, compartments))
        return result
    else:
        solver.delete()
        return None

# Q6
def find_c_alcuin_number(G: nx.Graph, c: int) -> int:
    """
    Compute the Alcuin_c(G) number for the given graph G and number of compartments c.

    Parameters:
    - G: A networkx.Graph representing the problem graph.
    - c: Un entier représentant le nombre de compartiments dans le bateau.

    Returns:
    - Le plus petit entier k tel qu'une séquence c-valide existe (c'est-à-dire Alcuin_c(G) = k).
      Retourne INFINITY si aucune solution n'existe.
    """
    n = len(G.nodes())
    # Le k maximal possible est n (le nombre de sujets)
    left = 1
    right = n
    alcuin_c_number = None

    # Recherche binaire
    while left <= right:
        mid = (left + right) // 2
        solution = gen_solution_cvalid(G, mid, c)
        if solution is not None:
            alcuin_c_number = mid
            right = mid - 1
        else:
            left = mid + 1

    # Si aucune solution n'a été trouvée, renvoyer INFINITY
    if alcuin_c_number is None:
        return float('inf')
    else:
        return alcuin_c_number



