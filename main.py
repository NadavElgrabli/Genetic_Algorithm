# Nadav Elgrabli ID 316082791
# בודק יקר, דיברתי עם צביקה המתרגל והוא אישר לי להגיש רק קובץ MAIN.PY ללא קובץ EXE עקב תקלות טכניות
# על מנת להריץ את התרגיל, יש לפתוח אותו בסביבת עבודה (עדיף PYCHARM) ולהריץ את main
# בתחילת התרגיל כתבתי קוד שאמור להוריד אוטומטית את כל הספריות הדרושות בשביל להריץ את הסימולציה ללא צורך בהתקנות ידניות
import subprocess
import sys
import pkg_resources

# This part downloads all the libraries automatically to the computer.
required_packages = {'matplotlib', 'numpy'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required_packages - installed
if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout = subprocess.DEVNULL)

import copy
import random
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

# Parameters
GENERATION_SIZE = 100
DUPLICATION_SCORE = 10
GREATER_THAN_RESTRICTION_SCORE = 10
SIZE = 0
MAX_GENERATIONS = 100
PRECENTAGE_OF_REPLICATION = 0.35


# Set a value for dict if not exists
def set_if_not_set(source, key, value):
    if key not in source:
        source[key] = value


# Create board from file
def create_board(file):
    global SIZE
    SIZE = int(file.readline())

    board = []
    for row_index in range(SIZE):
        row = []
        for column_index in range(SIZE):
            row.append(None)
        board.append(row)
    return board


# Set Pre-filled cells from file
def set_pre_filled_cells(file, board):
    num_of_pre_filled = int(file.readline())
    for cell in range(num_of_pre_filled):
        row, column, value = file.readline().split(" ")
        board[int(row) - 1][int(column) - 1] = int(value)


# Get restrictions array from file
def get_restrictions(file):
    restrictions = []
    num_of_restrictions = int(file.readline())
    for restriction in range(num_of_restrictions):
        row_gt, column_gt, row_lt, column_lt = file.readline().split(" ")
        restrictions.append((
            (int(row_gt) - 1, int(column_gt) - 1),
            (int(row_lt) - 1, int(column_lt) - 1)
        ))
    return restrictions


# Read board configuration file
def read_generations_file(file_name):
    with open(file_name) as file:
        board = create_board(file)
        set_pre_filled_cells(file, board)
        restrictions = get_restrictions(file)

        return board, restrictions


# Fill the board with values (where the cell is empty)
def fill_random_board(board):
    board_copy = copy.deepcopy(board)
    board_size = len(board_copy)

    for row in range(board_size):
        for column in range(board_size):
            if not board_copy[row][column]:
                board_copy[row][column] = random.randint(1, board_size)

    return board_copy


# Generate new generation (100 solutions)
def generate_generation(board):
    generation_solutions = []
    for generation_num in range(GENERATION_SIZE):
        generation_solutions.append(fill_random_board(board))
    return generation_solutions


# Count all the values in rows and columns
def count_board_values(board):
    board_size = len(board)
    count_dict = {}

    for row in range(board_size):
        for column in range(board_size):
            val = board[row][column]

            set_if_not_set(count_dict, f'row{row}', {})
            set_if_not_set(count_dict, f'column{column}', {})

            set_if_not_set(count_dict[f'row{row}'], val, 0)
            set_if_not_set(count_dict[f'column{column}'], val, 0)

            count_dict[f'row{row}'][val] += 1
            count_dict[f'column{column}'][val] += 1

    return count_dict


# Calculate the score of a given board, every mistake counts as 10 points. The lower the final score, the better the
# board.
def calculate_fitness(board, restrictions):
    count_dict = count_board_values(board)
    score = 0

    for gt, lt in restrictions:
        if board[gt[0]][gt[1]] <= board[lt[0]][lt[1]]:
            score += GREATER_THAN_RESTRICTION_SCORE

    for values_count in count_dict.values():
        for count in values_count.values():
            if count > 1:
                score += (count - 1) * DUPLICATION_SCORE

    return score

# Function that returns a random num between min and max values
def get_random_number(min, max):
    return random.randint(min, max)

# Function in charge of creating the next generation of boards.
def get_next_generation(generation, restrictions):
    with_scores = [(board, calculate_fitness(board, restrictions)) for board in generation]
    with_scores = sorted(with_scores, key=lambda tup: tup[1])

    # Copy the 35% best boards in the generation.
    best_boards = [tup[0] for tup in with_scores[:int(PRECENTAGE_OF_REPLICATION * len(generation))]]
    next_generation = copy.deepcopy(best_boards)
    assert None not in next_generation
    mutated_next_generation = copy.deepcopy(best_boards)
    assert None not in mutated_next_generation

    # Cross-over. Calculate 65% boards by applying random mutation to couples of boards from the copied population.
    while len(next_generation) < GENERATION_SIZE:
        parents = random.sample(best_boards, 2)
        child = create_child_board(parents[0], parents[1])
        next_generation.append(copy.deepcopy(child))
        mutated = mutate(copy.deepcopy(child), restrictions)
        mutated_next_generation.append(mutated)

        assert None not in mutated_next_generation

    return next_generation, mutated_next_generation

# Function returns the board with the best score in a specific generation
def best_in_generation(generation, restrictions):
    with_scores = [(board, calculate_fitness(board, restrictions)) for board in generation]
    with_scores = sorted(with_scores, key=lambda tup: tup[1])
    return with_scores[0]

# Function returns the average fitness score in a spcific generation
def avg_generation_score(generation, restrictions):
    scores = [calculate_fitness(board, restrictions) for board in generation]
    return np.average(scores)


# Mutation of changing a random number inside the board to a different random value.
def change_random_number(board):
    i = random.randint(0, SIZE - 1)
    j = random.randint(0, SIZE -1)
    k = random.randint(1, SIZE)

    board[i][j] = k
    return board

# Mutation for swapping between 2 random rows of board.
def shuffle_rows(board):
    i, j = random.sample(range(SIZE), 2)
    tmp = board[i]
    board[i] = board[j]
    board[j] = tmp
    return board

# Mutation function.
def mutate(board, restrictions):
    return change_random_number(board)
    #return shuffle_rows(board)
    #return lamarki_mutate(board, restrictions)

# Function makes sure the restrictions in the board are respected. If not, swap the numbers that
# don't respect the restriction
def swap_by_restriction(board, restrictions):
    for (i1, j1), (i2, j2) in restrictions:
        x = board[i1][j1]
        y = board[i2][j2]
        if x < y:
            board[i1][j1] = y
            board[i2][j2] = x
            return board, True

    return board, False

# lamarki mutation. Makes sure all restrictions are respected. If not, swap the numbers that dont respect.
# If all restrictions are respected it means there are numbers that appear more tahn once in certain rows / columns
# Threfore we use random mutation to improve the fitness score of the board.
def lamarki_mutate(board, restrictions):
    board, mutated = swap_by_restriction(board, restrictions)
    #if it all restrictions are respected, just change randomely a number in board.
    if not mutated:
        return change_random_number(board)
    else:
        return board

# Function in charge of dispalying a graph once we finished running through the amount of desired generations.
def show_graph(avg_fit, best_fit):
    assert len(avg_fit) == len(best_fit)
    plt.title(f'Average iteration score until generation number: {len(avg_fit)}')
    plt.plot(range(len(avg_fit)), avg_fit, best_fit)
    plt.ylabel('Fitness score')
    plt.legend(['Average fitness', 'Best fitness'])
    plt.xlabel('Generation number')
    plt.show()

# Function prints the current board.
def print_board(board, restrictions):
    size = len(board)
    for i in range(size):
        # Print row i.
        row_str = ''
        for j in range(size):
            row_str += str(board[i][j])
            if ((i, j), (i, j + 1)) in restrictions:
                row_str += '>'
            elif ((i, j + 1), (i, j)) in restrictions:
                row_str += '<'
            else:
                row_str += ' '
        print(row_str)
        # Print restrictions between row i and row i+1.
        row_str = ''
        for j in range(size):
            if ((i, j), (i + 1, j)) in restrictions:
                row_str += '> '
            elif ((i + 1, j), (i, j)) in restrictions:
                row_str += '< '
            else:
                row_str += '  '
        print(row_str)

# Function creates a child board from a given father and mother.
def create_child_board(mother_board, father_board):
    mother = mother_board
    father = father_board
    child_board = [[]] * SIZE

    # Select random row number
    row_index = random.choice(range(SIZE))

    # Copy all the rows until the random rows from mother to child
    child_board[row_index] = mother[row_index]

    # The rest of the rows that were not copied, copy from the father
    for i, row in enumerate(father):
        if i != row_index:
            child_board[i] = row
    return child_board


def main():

    # *IMPORTANT* Change to True if want to run darwinian algorithm.
    is_darwinian = False
    avg_scores = []
    best_scores = []

    # Read given file
    board, restrictions = read_generations_file('./generations.txt')

    last_generation = generate_generation(board)
    mutated_last_generation = copy.deepcopy(last_generation)

    best_board, best_score = best_in_generation(mutated_last_generation, restrictions)
    avg_score = avg_generation_score(mutated_last_generation, restrictions)

    generation_index = 0
    while generation_index < MAX_GENERATIONS:
        avg_scores.append(avg_score)
        best_scores.append(best_score)
        print(f"Generation {generation_index} best score: {best_score}")
        generation_index += 1

        if best_score == 0:
            break

        if is_darwinian:
            last_generation, mutated_last_generation = get_next_generation(copy.deepcopy(last_generation), restrictions)
        else:
            last_generation, mutated_last_generation = get_next_generation(copy.deepcopy(mutated_last_generation), restrictions)
        assert None not in last_generation
        assert None not in mutated_last_generation
        best_board, best_score = best_in_generation(mutated_last_generation, restrictions)
        avg_score = avg_generation_score(mutated_last_generation, restrictions)

    print("Best board:")
    print_board(best_board, restrictions)
    print(f"Best score: {best_score}, average score: {avg_score}")

    show_graph(avg_scores, best_scores)


if __name__ == "__main__":
   main()