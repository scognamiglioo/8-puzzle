import copy
import heapq
import time

class Puzzle:
    def __init__(self, size, start_state, goal_state):
        self.size = size
        self.start_state = start_state
        self.goal_state = goal_state

    def get_blank_position(self, state):
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] == 0:
                    return i, j

    def get_neighbors(self, state):
        blank_row, blank_col = self.get_blank_position(state)
        neighbors = []

        # Move blank space up
        if blank_row > 0:
            neighbor = copy.deepcopy(state)
            neighbor[blank_row][blank_col], neighbor[blank_row - 1][blank_col] = (
                neighbor[blank_row - 1][blank_col],
                neighbor[blank_row][blank_col],
            )
            neighbors.append(neighbor)

        # Move blank space down
        if blank_row < self.size - 1:
            neighbor = copy.deepcopy(state)
            neighbor[blank_row][blank_col], neighbor[blank_row + 1][blank_col] = (
                neighbor[blank_row + 1][blank_col],
                neighbor[blank_row][blank_col],
            )
            neighbors.append(neighbor)

        # Move blank space left
        if blank_col > 0:
            neighbor = copy.deepcopy(state)
            neighbor[blank_row][blank_col], neighbor[blank_row][blank_col - 1] = (
                neighbor[blank_row][blank_col - 1],
                neighbor[blank_row][blank_col],
            )
            neighbors.append(neighbor)

        # Move blank space right
        if blank_col < self.size - 1:
            neighbor = copy.deepcopy(state)
            neighbor[blank_row][blank_col], neighbor[blank_row][blank_col + 1] = (
                neighbor[blank_row][blank_col + 1],
                neighbor[blank_row][blank_col],
            )
            neighbors.append(neighbor)

        return neighbors

class Node:
    visited_nodes = 0

    def __init__(self, state, parent=None, cost=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        Node.visited_nodes += 1

    def __lt__(self, other):
        return self.cost < other.cost
    
    @classmethod
    def reset_visited_nodes(cls):
        cls.visited_nodes = 0

def uniform_cost_search(puzzle, visualize=False):
    Node.reset_visited_nodes()  # Reinicia a contagem antes de iniciar uma nova busca
    start_node = Node(state=puzzle.start_state, cost=0)
    open_set = [start_node]
    closed_set = set()
    start_time = time.time()

    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.state == puzzle.goal_state:
            end_time = time.time()
            elapsed_time = end_time - start_time
            solution_path = reconstruct_path(current_node)
            if visualize:
                visualize_solution(solution_path)
            return solution_path, elapsed_time, Node.visited_nodes

        closed_set.add(tuple(map(tuple, current_node.state)))

        for neighbor_state in puzzle.get_neighbors(current_node.state):
            if tuple(map(tuple, neighbor_state)) not in closed_set:
                neighbor_node = Node(state=neighbor_state, parent=current_node, cost=current_node.cost + 1)
                heapq.heappush(open_set, neighbor_node)

    return None, 0, Node.visited_nodes

def visualize_solution(solution_path):
    for i, state in enumerate(solution_path):
        print_puzzle(state, f'Step {i}')
        time.sleep(1)  # Ajuste o tempo conforme necessário

def print_puzzle(state, title):
    print(title)
    for row in state:
        print(row)
    print()
    time.sleep(1)  # Ajuste o tempo conforme necessário

def find_value_position(state, value):
    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] == value:
                return i, j

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

# Exemplo de uso:
size = 3
start_state = [[7, 2, 4], [5, 0, 6], [8, 3, 1]]
goal_state_complex = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

start_state_complex = [[7, 2, 4], [5, 0, 6], [8, 3, 1]]
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]


puzzle = Puzzle(size, start_state, goal_state)
solution_path, elapsed_time, visited_nodes = uniform_cost_search(puzzle, visualize=True)



with open('busca_uniforme.txt', 'w') as f:
    for state in solution_path:
        print_puzzle(state)
        f.write(str(state) + '\n')
    
    print(f'Número de passos: {len(solution_path) - 1}')
    print(f'Tempo decorrido: {elapsed_time} segundos')
    print(f'Número de nós visitados: {visited_nodes}')
    
    f.write(f'Número de passos: {len(solution_path) - 1}\n')
    f.write(f'Tempo decorrido: {elapsed_time} segundos\n')
    f.write(f'Número de nós visitados: {visited_nodes}')